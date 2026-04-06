#!/usr/bin/env python3
"""
Quantum-Enhanced LLM Cascade Routing: QAOA Proof of Concept
============================================================

Formulates the LLM model-selection problem as a QUBO and solves it using
Google Cirq's QAOA implementation, then benchmarks against classical solvers.

Problem: Given N tasks with complexity scores and M LLM models with
cost/quality profiles, find the minimum-cost assignment that satisfies
quality constraints and a total budget limit.

Usage:
    source ../venv/bin/activate
    python llm_routing_qaoa.py

Author: Amit Patole
"""

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cirq
import numpy as np
import scipy.optimize
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# ============================================================================
# 1. DATA MODEL — LLM Models and Task Profiles
# ============================================================================

@dataclass
class LLMModel:
    """An LLM model tier with cost and quality characteristics."""
    name: str
    cost_per_1k_tokens: float   # USD per 1K tokens
    quality_score: float         # 0.0 - 1.0 (capability rating)
    avg_latency_ms: int          # Average response latency
    max_tokens: int              # Context window

    def __repr__(self):
        return f"{self.name}(${self.cost_per_1k_tokens}/1K, q={self.quality_score})"


@dataclass
class Task:
    """A task to be routed to an LLM model."""
    task_id: str
    complexity: int              # 1=trivial, 2=simple, 3=moderate, 4=complex, 5=expert
    estimated_tokens: int        # Expected token consumption
    min_quality: float           # Minimum acceptable quality score
    sla_latency_ms: int          # Maximum acceptable latency

    def __repr__(self):
        return f"Task({self.task_id}, c={self.complexity}, ~{self.estimated_tokens}tok)"


@dataclass
class RoutingResult:
    """Result of a routing optimization."""
    assignments: dict            # task_id -> model_name
    total_cost: float
    quality_satisfied: float     # Fraction of tasks meeting quality threshold
    budget_satisfied: bool
    solve_time_ms: float
    method: str
    extra: dict = field(default_factory=dict)


# ============================================================================
# 2. PRODUCTION DATA — Real model tiers from Enginuity platform
# ============================================================================

# These match the actual cascade routing tiers in production
MODELS = [
    LLMModel("gemma2-2b",    0.000, 0.30, 50,   8192),    # Local Ollama — free
    LLMModel("haiku",        0.250, 0.60, 200,   200000),  # Claude Haiku
    LLMModel("sonnet",       3.000, 0.82, 500,   200000),  # Claude Sonnet
    LLMModel("opus",        15.000, 0.95, 1500,  200000),  # Claude Opus
    LLMModel("o1",          60.000, 0.99, 3000,  128000),  # OpenAI o1 (reference)
]

# Minimum quality required per complexity tier (from docs/61)
QUALITY_THRESHOLDS = {
    1: 0.20,  # trivial — any model works
    2: 0.40,  # simple — haiku+
    3: 0.70,  # moderate — sonnet+
    4: 0.85,  # complex — sonnet/opus
    5: 0.92,  # expert — opus+
}

# Latency SLA per complexity tier (ms)
LATENCY_SLA = {
    1: 5000,
    2: 10000,
    3: 30000,
    4: 60000,
    5: 120000,
}


def generate_task_batch(n_tasks: int, seed: int = 42) -> list[Task]:
    """Generate realistic task batches matching production complexity distribution.

    Distribution from production data (docs/61-resource-aware-optimization.md):
    - trivial (30%), simple (25%), moderate (25%), complex (15%), expert (5%)
    """
    rng = np.random.RandomState(seed)

    complexity_dist = [1]*30 + [2]*25 + [3]*25 + [4]*15 + [5]*5
    token_ranges = {
        1: (100, 500),
        2: (200, 1000),
        3: (500, 3000),
        4: (1000, 8000),
        5: (2000, 15000),
    }

    tasks = []
    for i in range(n_tasks):
        c = rng.choice(complexity_dist)
        tok_lo, tok_hi = token_ranges[c]
        tokens = int(rng.uniform(tok_lo, tok_hi))
        tasks.append(Task(
            task_id=f"T-{i:03d}",
            complexity=c,
            estimated_tokens=tokens,
            min_quality=QUALITY_THRESHOLDS[c],
            sla_latency_ms=LATENCY_SLA[c],
        ))
    return tasks


# ============================================================================
# 3. QUBO FORMULATION
# ============================================================================

def build_cost_matrix(tasks: list[Task], models: list[LLMModel]) -> np.ndarray:
    """Build the cost matrix C[i,j] = cost of assigning task i to model j."""
    n, m = len(tasks), len(models)
    C = np.zeros((n, m))
    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            C[i, j] = model.cost_per_1k_tokens * task.estimated_tokens / 1000.0
    return C


def build_quality_penalty(tasks: list[Task], models: list[LLMModel]) -> np.ndarray:
    """Build quality penalty matrix P[i,j] — penalty if model j can't meet task i's quality."""
    n, m = len(tasks), len(models)
    P = np.zeros((n, m))
    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            if model.quality_score < task.min_quality:
                P[i, j] = 10.0  # Large penalty for quality violation
    return P


def build_latency_penalty(tasks: list[Task], models: list[LLMModel]) -> np.ndarray:
    """Build latency penalty matrix L[i,j] — penalty if model j exceeds task i's SLA."""
    n, m = len(tasks), len(models)
    L = np.zeros((n, m))
    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            if model.avg_latency_ms > task.sla_latency_ms:
                L[i, j] = 5.0
    return L


def qubo_objective(x_flat: np.ndarray, tasks: list[Task], models: list[LLMModel],
                   budget: float, lambda_assign: float = 20.0,
                   lambda_quality: float = 15.0, lambda_budget: float = 10.0) -> float:
    """Evaluate the QUBO objective function for a given binary assignment vector.

    x_flat: Binary vector of length N*M, where x[i*M + j] = 1 means task i -> model j
    """
    n, m = len(tasks), len(models)
    X = x_flat.reshape(n, m)

    C = build_cost_matrix(tasks, models)
    P = build_quality_penalty(tasks, models)
    L = build_latency_penalty(tasks, models)

    # Cost term (minimize)
    cost = np.sum(X * C)

    # Assignment constraint: each task gets exactly one model
    assign_penalty = sum((1 - np.sum(X[i, :]))**2 for i in range(n))

    # Quality constraint
    quality_penalty = np.sum(X * P)

    # Budget constraint
    budget_penalty = max(0, cost - budget)**2

    # Latency constraint
    latency_penalty = np.sum(X * L)

    total = (cost
             + lambda_assign * assign_penalty
             + lambda_quality * quality_penalty
             + lambda_budget * budget_penalty
             + 2.0 * latency_penalty)

    return total


# ============================================================================
# 4. QAOA IMPLEMENTATION (Google Cirq)
# ============================================================================

class LLMRoutingQAOA:
    """QAOA solver for the LLM routing problem using Google Cirq."""

    def __init__(self, tasks: list[Task], models: list[LLMModel],
                 budget: float, p_layers: int = 2):
        self.tasks = tasks
        self.models = models
        self.budget = budget
        self.p = p_layers
        self.n = len(tasks)
        self.m = len(models)
        self.num_qubits = self.n * self.m

        # Create qubits — one per (task, model) pair
        self.qubits = cirq.LineQubit.range(self.num_qubits)

        # Precompute cost coefficients
        self.C = build_cost_matrix(tasks, models)
        self.P = build_quality_penalty(tasks, models)
        self.L = build_latency_penalty(tasks, models)

        # Combined linear coefficients for the Ising Hamiltonian
        self.h_coeffs = self._compute_linear_coefficients()
        self.J_coeffs = self._compute_coupling_coefficients()

    def _qubit_index(self, task_idx: int, model_idx: int) -> int:
        """Map (task, model) pair to qubit index."""
        return task_idx * self.m + model_idx

    def _compute_linear_coefficients(self) -> np.ndarray:
        """Compute linear (single-qubit) terms of the Ising Hamiltonian."""
        h = np.zeros(self.num_qubits)
        for i in range(self.n):
            for j in range(self.m):
                idx = self._qubit_index(i, j)
                # Cost contribution
                h[idx] += self.C[i, j]
                # Quality penalty
                h[idx] += 15.0 * self.P[i, j]
                # Latency penalty
                h[idx] += 2.0 * self.L[i, j]
        return h

    def _compute_coupling_coefficients(self) -> list[tuple[int, int, float]]:
        """Compute coupling (two-qubit) terms — enforce one-model-per-task constraint.

        For each task i, add repulsive coupling between all model pairs (j, k):
        if both x[i,j] and x[i,k] are 1, that violates the constraint.
        """
        couplings = []
        lambda_assign = 20.0
        for i in range(self.n):
            for j in range(self.m):
                for k in range(j + 1, self.m):
                    idx_j = self._qubit_index(i, j)
                    idx_k = self._qubit_index(i, k)
                    # Penalize both being 1 (double assignment)
                    couplings.append((idx_j, idx_k, lambda_assign))
        return couplings

    def build_cost_circuit(self, gamma: float) -> cirq.Circuit:
        """Build the cost/problem unitary: exp(-i * gamma * H_cost).

        Applies Z rotations for linear terms and ZZ interactions for couplings.
        """
        ops = []
        # Linear terms: Rz(2 * gamma * h_i) on each qubit
        for idx in range(self.num_qubits):
            if abs(self.h_coeffs[idx]) > 1e-10:
                angle = 2 * gamma * self.h_coeffs[idx]
                ops.append(cirq.rz(angle)(self.qubits[idx]))

        # Coupling terms: ZZ interactions
        for idx_a, idx_b, J_val in self.J_coeffs:
            angle = 2 * gamma * J_val
            ops.append(cirq.ZZPowGate(exponent=angle / np.pi)(
                self.qubits[idx_a], self.qubits[idx_b]))

        return cirq.Circuit(ops)

    def build_mixer_circuit(self, beta: float) -> cirq.Circuit:
        """Build the mixer unitary: exp(-i * beta * H_mixer).

        Standard X-mixer: Rx(2*beta) on each qubit.
        """
        ops = [cirq.rx(2 * beta)(q) for q in self.qubits]
        return cirq.Circuit(ops)

    def build_qaoa_circuit(self, gammas: list[float], betas: list[float]) -> cirq.Circuit:
        """Build the full QAOA circuit with p layers.

        |psi> = prod_{l=1}^{p} U_mixer(beta_l) U_cost(gamma_l) |+>^n
        """
        # Initial state: |+>^n (equal superposition)
        circuit = cirq.Circuit([cirq.H(q) for q in self.qubits])

        # p alternating layers
        for l in range(self.p):
            circuit += self.build_cost_circuit(gammas[l])
            circuit += self.build_mixer_circuit(betas[l])

        # Measurement
        circuit += cirq.Circuit([cirq.measure(*self.qubits, key='result')])

        return circuit

    def evaluate_assignment(self, bitstring: np.ndarray) -> float:
        """Evaluate the QUBO objective for a bitstring."""
        return qubo_objective(bitstring, self.tasks, self.models, self.budget)

    def solve(self, n_shots: int = 1000, n_restarts: int = 5,
              verbose: bool = False) -> RoutingResult:
        """Run QAOA optimization: find optimal (gamma, beta) parameters,
        sample the circuit, and return the best assignment found."""

        t0 = time.time()
        simulator = cirq.Simulator()

        best_cost = float('inf')
        best_bitstring = None
        best_params = None

        def objective(params):
            """Negative expectation value of the cost Hamiltonian."""
            gammas = params[:self.p]
            betas = params[self.p:]
            circuit = self.build_qaoa_circuit(list(gammas), list(betas))

            result = simulator.run(circuit, repetitions=n_shots)
            measurements = result.measurements['result']

            # Compute average cost over all samples
            costs = [self.evaluate_assignment(m) for m in measurements]
            return np.mean(costs)

        # Multiple random restarts for parameter optimization
        for restart in range(n_restarts):
            # Random initial parameters
            init_params = np.random.uniform(0, np.pi, 2 * self.p)

            if verbose:
                print(f"  QAOA restart {restart+1}/{n_restarts}...")

            # Optimize parameters using COBYLA (gradient-free)
            opt_result = scipy.optimize.minimize(
                objective, init_params,
                method='COBYLA',
                options={'maxiter': 50, 'rhobeg': 0.5}
            )

            # Sample the optimized circuit
            opt_gammas = opt_result.x[:self.p]
            opt_betas = opt_result.x[self.p:]
            final_circuit = self.build_qaoa_circuit(list(opt_gammas), list(opt_betas))
            final_result = simulator.run(final_circuit, repetitions=n_shots)
            measurements = final_result.measurements['result']

            # Find the best bitstring from samples
            for m in measurements:
                cost = self.evaluate_assignment(m)
                if cost < best_cost:
                    best_cost = cost
                    best_bitstring = m.copy()
                    best_params = opt_result.x.copy()

        solve_time = (time.time() - t0) * 1000

        # Decode the best assignment
        assignments = {}
        quality_met = 0
        total_cost = 0.0

        if best_bitstring is not None:
            X = best_bitstring.reshape(self.n, self.m)
            for i, task in enumerate(self.tasks):
                assigned = np.where(X[i] == 1)[0]
                if len(assigned) > 0:
                    j = assigned[0]  # Take first if multiple (constraint violation)
                    model = self.models[j]
                    assignments[task.task_id] = model.name
                    total_cost += model.cost_per_1k_tokens * task.estimated_tokens / 1000.0
                    if model.quality_score >= task.min_quality:
                        quality_met += 1
                else:
                    # No model assigned — fallback to cheapest valid
                    assignments[task.task_id] = "UNASSIGNED"

        return RoutingResult(
            assignments=assignments,
            total_cost=total_cost,
            quality_satisfied=quality_met / max(1, self.n),
            budget_satisfied=total_cost <= self.budget,
            solve_time_ms=solve_time,
            method=f"QAOA(p={self.p})",
            extra={
                "qubit_count": self.num_qubits,
                "circuit_depth": len(self.build_qaoa_circuit(
                    [0]*self.p, [0]*self.p)) if best_params is not None else 0,
                "optimal_params": best_params.tolist() if best_params is not None else [],
                "qubo_cost": best_cost,
                "n_restarts": n_restarts,
                "n_shots": n_shots,
            }
        )


# ============================================================================
# 5. CLASSICAL BASELINES
# ============================================================================

def solve_greedy(tasks: list[Task], models: list[LLMModel],
                 budget: float) -> RoutingResult:
    """Greedy solver — current production baseline.

    Assigns each task to the cheapest model that meets quality + latency constraints.
    This is what the Enginuity platform actually uses (complexity lookup table).
    """
    t0 = time.time()
    assignments = {}
    total_cost = 0.0
    quality_met = 0

    for task in tasks:
        best_model = None
        best_model_cost = float('inf')

        for model in models:
            if model.quality_score < task.min_quality:
                continue
            if model.avg_latency_ms > task.sla_latency_ms:
                continue
            cost = model.cost_per_1k_tokens * task.estimated_tokens / 1000.0
            if cost < best_model_cost:
                best_model = model
                best_model_cost = cost

        if best_model:
            assignments[task.task_id] = best_model.name
            total_cost += best_model_cost
            quality_met += 1
        else:
            # Fallback: cheapest model regardless
            cheapest = min(models, key=lambda m: m.cost_per_1k_tokens)
            assignments[task.task_id] = cheapest.name
            total_cost += cheapest.cost_per_1k_tokens * task.estimated_tokens / 1000.0

    solve_time = (time.time() - t0) * 1000

    return RoutingResult(
        assignments=assignments,
        total_cost=total_cost,
        quality_satisfied=quality_met / max(1, len(tasks)),
        budget_satisfied=total_cost <= budget,
        solve_time_ms=solve_time,
        method="Greedy (Production)",
    )


def solve_simulated_annealing(tasks: list[Task], models: list[LLMModel],
                               budget: float, n_iterations: int = 2000,
                               seed: int = 42) -> RoutingResult:
    """Simulated annealing solver for the LLM routing QUBO."""
    t0 = time.time()
    rng = np.random.RandomState(seed)
    n, m = len(tasks), len(models)

    # Initial solution: random valid assignment (one model per task)
    x = np.zeros(n * m, dtype=int)
    for i in range(n):
        j = rng.randint(0, m)
        x[i * m + j] = 1

    current_cost = qubo_objective(x, tasks, models, budget)
    best_x = x.copy()
    best_cost = current_cost

    # Annealing schedule
    T_init = 50.0
    T_final = 0.1

    for step in range(n_iterations):
        T = T_init * (T_final / T_init) ** (step / n_iterations)

        # Propose move: reassign one random task to a different model
        task_i = rng.randint(0, n)
        new_model_j = rng.randint(0, m)

        # Create neighbor
        x_new = x.copy()
        x_new[task_i * m: (task_i + 1) * m] = 0  # Clear current assignment
        x_new[task_i * m + new_model_j] = 1        # Assign new model

        new_cost = qubo_objective(x_new, tasks, models, budget)
        delta = new_cost - current_cost

        # Accept or reject
        if delta < 0 or rng.random() < np.exp(-delta / T):
            x = x_new
            current_cost = new_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()

    solve_time = (time.time() - t0) * 1000

    # Decode best solution
    X = best_x.reshape(n, m)
    assignments = {}
    total_cost = 0.0
    quality_met = 0

    for i, task in enumerate(tasks):
        assigned = np.where(X[i] == 1)[0]
        if len(assigned) > 0:
            j = assigned[0]
            model = models[j]
            assignments[task.task_id] = model.name
            total_cost += model.cost_per_1k_tokens * task.estimated_tokens / 1000.0
            if model.quality_score >= task.min_quality:
                quality_met += 1
        else:
            assignments[task.task_id] = "UNASSIGNED"

    return RoutingResult(
        assignments=assignments,
        total_cost=total_cost,
        quality_satisfied=quality_met / max(1, n),
        budget_satisfied=total_cost <= budget,
        solve_time_ms=solve_time,
        method="Simulated Annealing",
        extra={"n_iterations": n_iterations, "qubo_cost": best_cost},
    )


def solve_brute_force(tasks: list[Task], models: list[LLMModel],
                       budget: float) -> Optional[RoutingResult]:
    """Brute force solver — enumerate all possible assignments.

    Only feasible for very small instances (N*M <= 20).
    Provides the ground truth optimal solution.
    """
    n, m = len(tasks), len(models)
    total_combos = m ** n

    if total_combos > 1_000_000:
        print(f"  Brute force: {total_combos:,} combinations — skipping (too large)")
        return None

    t0 = time.time()
    best_cost = float('inf')
    best_assignment = None

    for combo in itertools.product(range(m), repeat=n):
        x = np.zeros(n * m, dtype=int)
        for i, j in enumerate(combo):
            x[i * m + j] = 1

        cost = qubo_objective(x, tasks, models, budget)
        if cost < best_cost:
            best_cost = cost
            best_assignment = combo

    solve_time = (time.time() - t0) * 1000

    # Decode
    assignments = {}
    total_cost = 0.0
    quality_met = 0
    for i, j in enumerate(best_assignment):
        task = tasks[i]
        model = models[j]
        assignments[task.task_id] = model.name
        total_cost += model.cost_per_1k_tokens * task.estimated_tokens / 1000.0
        if model.quality_score >= task.min_quality:
            quality_met += 1

    return RoutingResult(
        assignments=assignments,
        total_cost=total_cost,
        quality_satisfied=quality_met / max(1, n),
        budget_satisfied=total_cost <= budget,
        solve_time_ms=solve_time,
        method="Brute Force (Optimal)",
        extra={"total_combinations": total_combos, "qubo_cost": best_cost},
    )


# ============================================================================
# 6. EXPERIMENT RUNNER
# ============================================================================

def run_experiment(n_tasks: int, models: list[LLMModel], budget: float,
                   qaoa_layers: int = 2, seed: int = 42,
                   verbose: bool = True) -> dict:
    """Run a complete experiment: all solvers on the same problem instance."""

    tasks = generate_task_batch(n_tasks, seed=seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {n_tasks} tasks x {len(models)} models | Budget: ${budget:.2f}")
        print(f"{'='*70}")
        print(f"Tasks: {[t.complexity for t in tasks]}")
        print(f"Models: {[m.name for m in models]}")

    results = {}

    # 1. Greedy (production baseline)
    if verbose:
        print(f"\n[1/4] Running Greedy solver...")
    results['greedy'] = solve_greedy(tasks, models, budget)
    if verbose:
        r = results['greedy']
        print(f"  Cost: ${r.total_cost:.4f} | Quality: {r.quality_satisfied:.0%} | "
              f"Time: {r.solve_time_ms:.1f}ms")

    # 2. Simulated Annealing
    if verbose:
        print(f"\n[2/4] Running Simulated Annealing...")
    results['sa'] = solve_simulated_annealing(tasks, models, budget, seed=seed)
    if verbose:
        r = results['sa']
        print(f"  Cost: ${r.total_cost:.4f} | Quality: {r.quality_satisfied:.0%} | "
              f"Time: {r.solve_time_ms:.1f}ms")

    # 3. Brute Force (small instances only)
    if verbose:
        print(f"\n[3/4] Running Brute Force (if feasible)...")
    results['brute_force'] = solve_brute_force(tasks, models, budget)
    if results['brute_force'] and verbose:
        r = results['brute_force']
        print(f"  Cost: ${r.total_cost:.4f} | Quality: {r.quality_satisfied:.0%} | "
              f"Time: {r.solve_time_ms:.1f}ms | Combos: {r.extra['total_combinations']:,}")

    # 4. QAOA
    if verbose:
        print(f"\n[4/4] Running QAOA (p={qaoa_layers}, {n_tasks * len(models)} qubits)...")
    qaoa = LLMRoutingQAOA(tasks, models, budget, p_layers=qaoa_layers)
    results['qaoa'] = qaoa.solve(n_shots=500, n_restarts=3, verbose=verbose)
    if verbose:
        r = results['qaoa']
        print(f"  Cost: ${r.total_cost:.4f} | Quality: {r.quality_satisfied:.0%} | "
              f"Time: {r.solve_time_ms:.1f}ms | Qubits: {r.extra['qubit_count']}")

    return {
        'n_tasks': n_tasks,
        'n_models': len(models),
        'budget': budget,
        'results': results,
        'tasks': tasks,
    }


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def plot_results(experiments: list[dict], output_dir: str = "../results"):
    """Generate publication-quality comparison plots."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Cost Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = ['greedy', 'sa', 'qaoa']
    method_labels = ['Greedy\n(Production)', 'Simulated\nAnnealing', 'QAOA\n(Quantum)']
    colors = ['#2196F3', '#FF9800', '#9C27B0']

    # 1a: Total cost by method
    ax = axes[0]
    for exp in experiments:
        n = exp['n_tasks']
        costs = [exp['results'][m].total_cost if exp['results'].get(m) else 0
                 for m in methods]
        x = np.arange(len(methods))
        ax.bar(x + 0.2 * experiments.index(exp), costs,
               width=0.2, label=f"N={n}")
    ax.set_xticks(np.arange(len(methods)) + 0.1 * (len(experiments) - 1))
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel('Total Cost (USD)')
    ax.set_title('(a) Total Routing Cost')
    ax.legend()

    # 1b: Quality satisfaction
    ax = axes[1]
    for exp in experiments:
        n = exp['n_tasks']
        quality = [exp['results'][m].quality_satisfied * 100 if exp['results'].get(m) else 0
                   for m in methods]
        x = np.arange(len(methods))
        ax.bar(x + 0.2 * experiments.index(exp), quality,
               width=0.2, label=f"N={n}")
    ax.set_xticks(np.arange(len(methods)) + 0.1 * (len(experiments) - 1))
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel('Quality Satisfaction (%)')
    ax.set_title('(b) Quality Constraint Satisfaction')
    ax.set_ylim(0, 110)
    ax.legend()

    # 1c: Solve time
    ax = axes[2]
    for exp in experiments:
        n = exp['n_tasks']
        times = [exp['results'][m].solve_time_ms if exp['results'].get(m) else 0
                 for m in methods]
        x = np.arange(len(methods))
        ax.bar(x + 0.2 * experiments.index(exp), times,
               width=0.2, label=f"N={n}")
    ax.set_xticks(np.arange(len(methods)) + 0.1 * (len(experiments) - 1))
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel('Solve Time (ms)')
    ax.set_title('(c) Computation Time')
    ax.set_yscale('log')
    ax.legend()

    plt.suptitle('Quantum-Enhanced LLM Cascade Routing: QAOA vs Classical',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_cost_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/fig1_cost_comparison.png")

    # --- Plot 2: Approximation Ratio (QAOA vs Optimal for small instances) ---
    small_exps = [e for e in experiments
                  if e['results'].get('brute_force') is not None]

    if small_exps:
        fig, ax = plt.subplots(figsize=(8, 5))
        for exp in small_exps:
            optimal_cost = exp['results']['brute_force'].extra.get('qubo_cost', 1)
            if optimal_cost > 0:
                ratios = {}
                for m in methods:
                    if exp['results'].get(m):
                        r_cost = exp['results'][m].extra.get('qubo_cost',
                                    qubo_objective(
                                        np.zeros(exp['n_tasks'] * exp['n_models']),
                                        exp['tasks'], MODELS[:exp['n_models']],
                                        exp['budget']))
                        ratios[m] = r_cost / optimal_cost

                x = np.arange(len(methods))
                bars = [ratios.get(m, 0) for m in methods]
                ax.bar(x, bars, color=colors)
                ax.axhline(y=1.0, color='red', linestyle='--', label='Optimal')

        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(method_labels, fontsize=10)
        ax.set_ylabel('Approximation Ratio (lower = better)')
        ax.set_title('QUBO Objective: Approximation Ratio vs Optimal')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig2_approximation_ratio.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/fig2_approximation_ratio.png")

    # --- Plot 3: Assignment Heatmap (for one experiment) ---
    exp = experiments[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, method in enumerate(['greedy', 'sa', 'qaoa']):
        ax = axes[idx]
        r = exp['results'].get(method)
        if r is None:
            continue

        model_names = [m.name for m in MODELS[:exp['n_models']]]
        heatmap = np.zeros((exp['n_tasks'], exp['n_models']))
        for i, task in enumerate(exp['tasks']):
            assigned_model = r.assignments.get(task.task_id, '')
            for j, mn in enumerate(model_names):
                if assigned_model == mn:
                    heatmap[i, j] = task.complexity

        im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('Model')
        ax.set_ylabel('Task')
        ax.set_xticks(range(exp['n_models']))
        ax.set_xticklabels(model_names, rotation=45, fontsize=8)
        ax.set_title(f"{method_labels[idx]}\nCost: ${r.total_cost:.4f}")

    plt.suptitle('Task-to-Model Assignment Heatmap (color = complexity)',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes, label='Task Complexity', shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_assignment_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig3_assignment_heatmap.png")

    plt.close('all')


def print_summary_table(experiments: list[dict]):
    """Print a publication-ready summary table."""
    print(f"\n{'='*90}")
    print(f"SUMMARY TABLE — Quantum-Enhanced LLM Cascade Routing Results")
    print(f"{'='*90}")
    print(f"{'Problem':>12} | {'Method':>22} | {'Cost ($)':>10} | {'Quality':>8} | "
          f"{'Time (ms)':>10} | {'Qubits':>7}")
    print(f"{'-'*12}-+-{'-'*22}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}")

    for exp in experiments:
        label = f"{exp['n_tasks']}T x {exp['n_models']}M"
        for method_key, method_name in [('brute_force', 'Brute Force (Optimal)'),
                                         ('greedy', 'Greedy (Production)'),
                                         ('sa', 'Simulated Annealing'),
                                         ('qaoa', 'QAOA (Quantum)')]:
            r = exp['results'].get(method_key)
            if r is None:
                continue
            qubits = r.extra.get('qubit_count', '-')
            print(f"{label:>12} | {r.method:>22} | ${r.total_cost:>8.4f} | "
                  f"{r.quality_satisfied:>7.0%} | {r.solve_time_ms:>9.1f} | "
                  f"{str(qubits):>7}")
        print(f"{'-'*12}-+-{'-'*22}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}")


# ============================================================================
# 8. MAIN — Run all experiments
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Quantum-Enhanced LLM Cascade Routing — QAOA Proof of Concept")
    print("Using Google Cirq Simulator")
    print("=" * 70)
    print(f"\nCirq version: {cirq.__version__}")
    print(f"NumPy version: {np.__version__}")

    experiments = []

    # Experiment 1: Small instance (brute-force solvable for ground truth)
    # 4 tasks x 3 models = 12 qubits, 81 combinations
    exp1 = run_experiment(
        n_tasks=4,
        models=MODELS[:3],  # gemma2, haiku, sonnet only
        budget=5.0,
        qaoa_layers=2,
        seed=42,
    )
    experiments.append(exp1)

    # Experiment 2: Medium instance
    # 8 tasks x 4 models = 32 qubits
    exp2 = run_experiment(
        n_tasks=8,
        models=MODELS[:4],  # gemma2, haiku, sonnet, opus
        budget=20.0,
        qaoa_layers=2,
        seed=42,
    )
    experiments.append(exp2)

    # Experiment 3: Larger instance (production-like)
    # 6 tasks x 5 models = 30 qubits
    exp3 = run_experiment(
        n_tasks=6,
        models=MODELS,  # All 5 models
        budget=50.0,
        qaoa_layers=3,
        seed=42,
    )
    experiments.append(exp3)

    # Summary
    print_summary_table(experiments)

    # Generate plots
    plot_results(experiments)

    # Save raw results as JSON
    results_json = []
    for exp in experiments:
        exp_data = {
            'n_tasks': exp['n_tasks'],
            'n_models': exp['n_models'],
            'budget': exp['budget'],
            'results': {}
        }
        for key, r in exp['results'].items():
            if r is not None:
                exp_data['results'][key] = {
                    'method': r.method,
                    'total_cost': r.total_cost,
                    'quality_satisfied': r.quality_satisfied,
                    'budget_satisfied': r.budget_satisfied,
                    'solve_time_ms': r.solve_time_ms,
                    'assignments': r.assignments,
                    'extra': r.extra,
                }
        results_json.append(exp_data)

    results_path = Path("../results/experiment_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved raw results: {results_path}")

    print("\n" + "=" * 70)
    print("DONE. Results and figures saved to ../results/")
    print("=" * 70)
