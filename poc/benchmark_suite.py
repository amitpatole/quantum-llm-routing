#!/usr/bin/env python3
"""
Benchmark Suite — Quantum vs Classical LLM Routing
====================================================

Extended experiments with:
1. Tuned QAOA penalty weights (quality-aware)
2. Multiple QAOA depth (p=1,2,3,4) comparison
3. Scaling analysis (vary N tasks, M models)
4. Approximation ratio vs brute-force optimal
5. Cost-quality Pareto frontier analysis
6. Publication-ready figures and tables

Usage:
    source ../venv/bin/activate
    python benchmark_suite.py

Author: Amit Patole
"""

import json
import time
from pathlib import Path

import cirq
import numpy as np
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from llm_routing_qaoa import (
    MODELS, QUALITY_THRESHOLDS, LATENCY_SLA,
    LLMModel, Task, RoutingResult,
    generate_task_batch, build_cost_matrix,
    build_quality_penalty, build_latency_penalty,
    solve_greedy, solve_simulated_annealing, solve_brute_force,
)


# ============================================================================
# 1. IMPROVED QAOA — Stronger constraint penalties
# ============================================================================

def qubo_objective_v2(x_flat, tasks, models, budget,
                      lambda_assign=50.0, lambda_quality=40.0,
                      lambda_budget=5.0, lambda_latency=3.0):
    """Improved QUBO with stronger quality/assignment penalties.

    Key change: lambda_quality=40 (was 15) forces the optimizer to
    respect quality constraints rather than defaulting to cheapest model.
    """
    n, m = len(tasks), len(models)
    X = x_flat.reshape(n, m)

    C = build_cost_matrix(tasks, models)
    P = build_quality_penalty(tasks, models)
    L = build_latency_penalty(tasks, models)

    # Normalize cost to [0, 1] range to balance with penalties
    max_cost = np.max(C) if np.max(C) > 0 else 1.0
    C_norm = C / max_cost

    cost = np.sum(X * C_norm)
    assign_penalty = sum((1 - np.sum(X[i, :]))**2 for i in range(n))
    quality_penalty = np.sum(X * P)
    total_real_cost = np.sum(X * C)
    budget_penalty = max(0, total_real_cost - budget)**2 / (budget**2) if budget > 0 else 0
    latency_penalty = np.sum(X * L)

    total = (cost
             + lambda_assign * assign_penalty
             + lambda_quality * quality_penalty
             + lambda_budget * budget_penalty
             + lambda_latency * latency_penalty)
    return total


class ImprovedQAOA:
    """QAOA solver with tuned penalties and multiple optimization strategies."""

    def __init__(self, tasks, models, budget, p_layers=2):
        self.tasks = tasks
        self.models = models
        self.budget = budget
        self.p = p_layers
        self.n = len(tasks)
        self.m = len(models)
        self.num_qubits = self.n * self.m
        self.qubits = cirq.LineQubit.range(self.num_qubits)

        self.C = build_cost_matrix(tasks, models)
        self.P = build_quality_penalty(tasks, models)
        self.L = build_latency_penalty(tasks, models)
        self.h_coeffs = self._compute_h()
        self.J_coeffs = self._compute_J()

    def _qi(self, i, j):
        return i * self.m + j

    def _compute_h(self):
        """Linear Ising terms with improved penalty scaling."""
        h = np.zeros(self.num_qubits)
        max_cost = np.max(self.C) if np.max(self.C) > 0 else 1.0
        for i in range(self.n):
            for j in range(self.m):
                idx = self._qi(i, j)
                h[idx] += self.C[i, j] / max_cost        # Normalized cost
                h[idx] += 40.0 * self.P[i, j]            # Strong quality penalty
                h[idx] += 3.0 * self.L[i, j]             # Latency penalty
        return h

    def _compute_J(self):
        """Two-qubit coupling terms — strong one-hot constraint."""
        couplings = []
        for i in range(self.n):
            for j in range(self.m):
                for k in range(j + 1, self.m):
                    couplings.append((self._qi(i, j), self._qi(i, k), 50.0))
        return couplings

    def build_circuit(self, gammas, betas):
        circuit = cirq.Circuit([cirq.H(q) for q in self.qubits])
        for l in range(self.p):
            # Cost layer
            ops = []
            for idx in range(self.num_qubits):
                if abs(self.h_coeffs[idx]) > 1e-10:
                    ops.append(cirq.rz(2 * gammas[l] * self.h_coeffs[idx])(self.qubits[idx]))
            for a, b, J in self.J_coeffs:
                ops.append(cirq.ZZPowGate(exponent=2 * gammas[l] * J / np.pi)(
                    self.qubits[a], self.qubits[b]))
            circuit += cirq.Circuit(ops)
            # Mixer layer
            circuit += cirq.Circuit([cirq.rx(2 * betas[l])(q) for q in self.qubits])
        circuit += cirq.Circuit([cirq.measure(*self.qubits, key='r')])
        return circuit

    def solve(self, n_shots=1000, n_restarts=5, verbose=False):
        t0 = time.time()
        simulator = cirq.Simulator()
        best_cost = float('inf')
        best_bits = None

        def objective(params):
            gammas, betas = params[:self.p], params[self.p:]
            circ = self.build_circuit(list(gammas), list(betas))
            res = simulator.run(circ, repetitions=n_shots)
            meas = res.measurements['r']
            costs = [qubo_objective_v2(m, self.tasks, self.models, self.budget) for m in meas]
            return np.mean(costs)

        for restart in range(n_restarts):
            if verbose:
                print(f"  QAOA-v2 restart {restart+1}/{n_restarts}...")
            init = np.random.uniform(0, np.pi, 2 * self.p)
            opt = scipy.optimize.minimize(objective, init, method='COBYLA',
                                          options={'maxiter': 60, 'rhobeg': 0.5})
            circ = self.build_circuit(list(opt.x[:self.p]), list(opt.x[self.p:]))
            res = simulator.run(circ, repetitions=n_shots)
            meas = res.measurements['r']
            for m in meas:
                c = qubo_objective_v2(m, self.tasks, self.models, self.budget)
                if c < best_cost:
                    best_cost = c
                    best_bits = m.copy()

        solve_time = (time.time() - t0) * 1000

        # Decode
        assignments = {}
        total_cost = 0.0
        quality_met = 0
        if best_bits is not None:
            X = best_bits.reshape(self.n, self.m)
            for i, task in enumerate(self.tasks):
                assigned = np.where(X[i] == 1)[0]
                if len(assigned) > 0:
                    j = assigned[0]
                    model = self.models[j]
                    assignments[task.task_id] = model.name
                    total_cost += model.cost_per_1k_tokens * task.estimated_tokens / 1000
                    if model.quality_score >= task.min_quality:
                        quality_met += 1
                else:
                    assignments[task.task_id] = "UNASSIGNED"

        return RoutingResult(
            assignments=assignments,
            total_cost=total_cost,
            quality_satisfied=quality_met / max(1, self.n),
            budget_satisfied=total_cost <= self.budget,
            solve_time_ms=solve_time,
            method=f"QAOA-v2(p={self.p})",
            extra={'qubit_count': self.num_qubits, 'qubo_cost': best_cost,
                   'n_restarts': n_restarts, 'n_shots': n_shots},
        )


# ============================================================================
# 2. EXTENDED EXPERIMENTS
# ============================================================================

def experiment_penalty_tuning():
    """Show how penalty weight affects QAOA quality satisfaction.

    This is a key paper contribution — demonstrates the constraint
    satisfaction vs cost minimization tradeoff in QUBO formulations.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Penalty Weight Sensitivity Analysis")
    print("=" * 70)

    tasks = generate_task_batch(4, seed=42)
    models = MODELS[:3]
    budget = 5.0

    lambda_values = [5, 10, 20, 40, 80]
    results = []

    for lam_q in lambda_values:
        print(f"\n  lambda_quality = {lam_q}...")

        # Custom QAOA with this penalty weight
        qaoa = ImprovedQAOA(tasks, models, budget, p_layers=2)
        # Override the quality penalty weight
        for idx in range(qaoa.num_qubits):
            i = idx // qaoa.m
            j = idx % qaoa.m
            # Reset and recompute with new lambda
            qaoa.h_coeffs[idx] = (qaoa.C[i, j] / (np.max(qaoa.C) or 1)
                                  + lam_q * qaoa.P[i, j]
                                  + 3.0 * qaoa.L[i, j])

        r = qaoa.solve(n_shots=500, n_restarts=3)
        results.append({
            'lambda_quality': lam_q,
            'cost': r.total_cost,
            'quality': r.quality_satisfied,
            'method': r.method,
        })
        print(f"    Cost: ${r.total_cost:.4f} | Quality: {r.quality_satisfied:.0%}")

    return results


def experiment_qaoa_depth():
    """Compare QAOA performance across circuit depths p=1,2,3,4.

    Shows how more QAOA layers improve solution quality (with
    diminishing returns), at the cost of longer optimization time.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: QAOA Circuit Depth Analysis (p=1,2,3,4)")
    print("=" * 70)

    tasks = generate_task_batch(4, seed=42)
    models = MODELS[:3]
    budget = 5.0

    results = []
    for p in [1, 2, 3, 4]:
        print(f"\n  p={p} layers ({4 * 3} qubits, depth ~{p * 2 + 1})...")
        qaoa = ImprovedQAOA(tasks, models, budget, p_layers=p)
        r = qaoa.solve(n_shots=500, n_restarts=3, verbose=True)
        results.append({
            'p': p,
            'cost': r.total_cost,
            'quality': r.quality_satisfied,
            'time_ms': r.solve_time_ms,
            'qubo_cost': r.extra['qubo_cost'],
        })
        print(f"    Cost: ${r.total_cost:.4f} | Quality: {r.quality_satisfied:.0%} | "
              f"Time: {r.solve_time_ms:.0f}ms")

    return results


def experiment_scaling():
    """Measure how problem size affects each solver's performance.

    Varies N (tasks) while keeping M=3 models fixed.
    Shows QAOA qubit requirements and solve time growth.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Scaling Analysis (N=2,4,6,8 tasks)")
    print("=" * 70)

    m = 3
    models = MODELS[:m]
    task_counts = [2, 4, 6, 8]
    results = []

    for n in task_counts:
        print(f"\n  N={n} tasks x {m} models = {n*m} qubits...")
        tasks = generate_task_batch(n, seed=42)
        budget = n * 5.0

        row = {'n_tasks': n, 'n_qubits': n * m}

        # Greedy
        r = solve_greedy(tasks, models, budget)
        row['greedy_cost'] = r.total_cost
        row['greedy_quality'] = r.quality_satisfied
        row['greedy_time'] = r.solve_time_ms

        # SA
        r = solve_simulated_annealing(tasks, models, budget)
        row['sa_cost'] = r.total_cost
        row['sa_quality'] = r.quality_satisfied
        row['sa_time'] = r.solve_time_ms

        # QAOA-v2
        qaoa = ImprovedQAOA(tasks, models, budget, p_layers=2)
        r = qaoa.solve(n_shots=500, n_restarts=3, verbose=True)
        row['qaoa_cost'] = r.total_cost
        row['qaoa_quality'] = r.quality_satisfied
        row['qaoa_time'] = r.solve_time_ms

        # Brute force (if feasible)
        bf = solve_brute_force(tasks, models, budget)
        if bf:
            row['optimal_cost'] = bf.total_cost
            row['optimal_quality'] = bf.quality_satisfied
            row['optimal_time'] = bf.solve_time_ms
        else:
            row['optimal_cost'] = None

        results.append(row)
        print(f"    Greedy: ${row['greedy_cost']:.4f} ({row['greedy_quality']:.0%}) | "
              f"SA: ${row['sa_cost']:.4f} ({row['sa_quality']:.0%}) | "
              f"QAOA: ${row['qaoa_cost']:.4f} ({row['qaoa_quality']:.0%})")

    return results


def experiment_pareto():
    """Generate cost-quality Pareto frontier for each solver.

    Varies the budget constraint and plots the frontier of achievable
    (cost, quality) pairs — reveals each solver's tradeoff behavior.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Cost-Quality Pareto Frontier")
    print("=" * 70)

    tasks = generate_task_batch(4, seed=42)
    models = MODELS[:3]
    budgets = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    results = {'greedy': [], 'sa': [], 'qaoa': []}

    for b in budgets:
        print(f"\n  Budget: ${b:.2f}...")

        r = solve_greedy(tasks, models, b)
        results['greedy'].append((r.total_cost, r.quality_satisfied))

        r = solve_simulated_annealing(tasks, models, b)
        results['sa'].append((r.total_cost, r.quality_satisfied))

        qaoa = ImprovedQAOA(tasks, models, b, p_layers=2)
        r = qaoa.solve(n_shots=500, n_restarts=3)
        results['qaoa'].append((r.total_cost, r.quality_satisfied))

        print(f"    G: ${results['greedy'][-1][0]:.2f}/{results['greedy'][-1][1]:.0%} | "
              f"SA: ${results['sa'][-1][0]:.2f}/{results['sa'][-1][1]:.0%} | "
              f"Q: ${results['qaoa'][-1][0]:.2f}/{results['qaoa'][-1][1]:.0%}")

    return results


# ============================================================================
# 3. PUBLICATION FIGURES
# ============================================================================

def plot_penalty_sensitivity(results, output_dir):
    """Figure: How quality penalty weight affects QAOA solution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    lambdas = [r['lambda_quality'] for r in results]
    costs = [r['cost'] for r in results]
    qualities = [r['quality'] * 100 for r in results]

    ax1.plot(lambdas, costs, 'o-', color='#9C27B0', linewidth=2, markersize=8)
    ax1.set_xlabel(r'$\lambda_{quality}$ (penalty weight)', fontsize=11)
    ax1.set_ylabel('Total Cost (USD)', fontsize=11)
    ax1.set_title('(a) Cost vs Quality Penalty Weight')
    ax1.grid(alpha=0.3)

    ax2.plot(lambdas, qualities, 's-', color='#4CAF50', linewidth=2, markersize=8)
    ax2.set_xlabel(r'$\lambda_{quality}$ (penalty weight)', fontsize=11)
    ax2.set_ylabel('Quality Satisfaction (%)', fontsize=11)
    ax2.set_title('(b) Quality Satisfaction vs Penalty Weight')
    ax2.set_ylim(0, 110)
    ax2.grid(alpha=0.3)

    plt.suptitle('QAOA Penalty Weight Sensitivity Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_penalty_sensitivity.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig4_penalty_sensitivity.png")
    plt.close()


def plot_depth_analysis(results, output_dir):
    """Figure: QAOA performance vs circuit depth."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    ps = [r['p'] for r in results]
    costs = [r['cost'] for r in results]
    qualities = [r['quality'] * 100 for r in results]
    times = [r['time_ms'] for r in results]

    ax1.bar(ps, costs, color='#9C27B0', alpha=0.8)
    ax1.set_xlabel('QAOA Layers (p)', fontsize=11)
    ax1.set_ylabel('Total Cost (USD)', fontsize=11)
    ax1.set_title('(a) Routing Cost')
    ax1.set_xticks(ps)

    ax2.bar(ps, qualities, color='#4CAF50', alpha=0.8)
    ax2.set_xlabel('QAOA Layers (p)', fontsize=11)
    ax2.set_ylabel('Quality Satisfaction (%)', fontsize=11)
    ax2.set_title('(b) Quality Satisfaction')
    ax2.set_xticks(ps)
    ax2.set_ylim(0, 110)

    ax3.bar(ps, times, color='#FF9800', alpha=0.8)
    ax3.set_xlabel('QAOA Layers (p)', fontsize=11)
    ax3.set_ylabel('Solve Time (ms)', fontsize=11)
    ax3.set_title('(c) Computation Time')
    ax3.set_xticks(ps)

    plt.suptitle('Effect of QAOA Circuit Depth on Solution Quality', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_depth_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig5_depth_analysis.png")
    plt.close()


def plot_scaling(results, output_dir):
    """Figure: How solvers scale with problem size."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    ns = [r['n_tasks'] for r in results]

    # Cost comparison
    ax1.plot(ns, [r['greedy_cost'] for r in results], 'o-', label='Greedy', color='#2196F3', linewidth=2)
    ax1.plot(ns, [r['sa_cost'] for r in results], 's-', label='Sim. Annealing', color='#FF9800', linewidth=2)
    ax1.plot(ns, [r['qaoa_cost'] for r in results], '^-', label='QAOA-v2', color='#9C27B0', linewidth=2)
    opt_ns = [r['n_tasks'] for r in results if r.get('optimal_cost') is not None]
    opt_costs = [r['optimal_cost'] for r in results if r.get('optimal_cost') is not None]
    if opt_ns:
        ax1.plot(opt_ns, opt_costs, 'D--', label='Optimal', color='red', linewidth=1.5)
    ax1.set_xlabel('Number of Tasks (N)', fontsize=11)
    ax1.set_ylabel('Total Cost (USD)', fontsize=11)
    ax1.set_title('(a) Cost Scaling')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Quality comparison
    ax2.plot(ns, [r['greedy_quality']*100 for r in results], 'o-', label='Greedy', color='#2196F3', linewidth=2)
    ax2.plot(ns, [r['sa_quality']*100 for r in results], 's-', label='Sim. Annealing', color='#FF9800', linewidth=2)
    ax2.plot(ns, [r['qaoa_quality']*100 for r in results], '^-', label='QAOA-v2', color='#9C27B0', linewidth=2)
    ax2.set_xlabel('Number of Tasks (N)', fontsize=11)
    ax2.set_ylabel('Quality Satisfaction (%)', fontsize=11)
    ax2.set_title('(b) Quality Scaling')
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Time comparison
    ax3.plot(ns, [r['greedy_time'] for r in results], 'o-', label='Greedy', color='#2196F3', linewidth=2)
    ax3.plot(ns, [r['sa_time'] for r in results], 's-', label='Sim. Annealing', color='#FF9800', linewidth=2)
    ax3.plot(ns, [r['qaoa_time'] for r in results], '^-', label='QAOA-v2', color='#9C27B0', linewidth=2)
    ax3.set_xlabel('Number of Tasks (N)', fontsize=11)
    ax3.set_ylabel('Solve Time (ms)', fontsize=11)
    ax3.set_title('(c) Time Scaling')
    ax3.set_yscale('log')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.suptitle('Solver Performance Scaling with Problem Size', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig6_scaling.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig6_scaling.png")
    plt.close()


def plot_pareto(results, output_dir):
    """Figure: Cost-quality Pareto frontier for each solver."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'greedy': '#2196F3', 'sa': '#FF9800', 'qaoa': '#9C27B0'}
    labels = {'greedy': 'Greedy (Production)', 'sa': 'Simulated Annealing', 'qaoa': 'QAOA-v2'}
    markers = {'greedy': 'o', 'sa': 's', 'qaoa': '^'}

    for method in ['greedy', 'sa', 'qaoa']:
        costs = [p[0] for p in results[method]]
        quals = [p[1] * 100 for p in results[method]]
        ax.scatter(costs, quals, c=colors[method], marker=markers[method],
                   s=80, label=labels[method], zorder=5)
        # Connect points
        sorted_pairs = sorted(zip(costs, quals))
        ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
                color=colors[method], alpha=0.4, linewidth=1.5)

    # Ideal region
    ax.axhspan(80, 110, alpha=0.08, color='green', label='High Quality Zone (>80%)')
    ax.axvspan(0, 5, alpha=0.08, color='blue', label='Low Cost Zone (<$5)')

    ax.set_xlabel('Total Routing Cost (USD)', fontsize=12)
    ax.set_ylabel('Quality Satisfaction (%)', fontsize=12)
    ax.set_title('Cost-Quality Pareto Frontier: Quantum vs Classical', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig7_pareto_frontier.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig7_pareto_frontier.png")
    plt.close()


def plot_circuit_diagram(output_dir):
    """Generate a QAOA circuit diagram for the paper."""
    # Small example: 2 tasks x 2 models = 4 qubits, p=1
    tasks = generate_task_batch(2, seed=42)[:2]
    models = MODELS[:2]

    qaoa = ImprovedQAOA(tasks, models, budget=5.0, p_layers=1)
    circuit = qaoa.build_circuit([0.5], [0.3])

    # Remove measurement for cleaner diagram
    circuit_no_meas = cirq.Circuit([op for mom in circuit for op in mom
                                    if not isinstance(op.gate, cirq.MeasurementGate)])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.text(0.5, 0.5, str(circuit_no_meas), transform=ax.transAxes,
            fontsize=8, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('QAOA Circuit for LLM Routing (2 tasks x 2 models, p=1)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig8_circuit_diagram.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig8_circuit_diagram.png")
    plt.close()


def generate_latex_table(scaling_results):
    """Generate LaTeX table for the paper."""
    print("\n" + "=" * 70)
    print("LaTeX Table (paste into paper)")
    print("=" * 70)
    print(r"""
\begin{table}[h]
\centering
\caption{Solver Performance Comparison Across Problem Sizes}
\label{tab:results}
\begin{tabular}{l|cc|cc|cc|cc}
\hline
\textbf{Size} & \multicolumn{2}{c|}{\textbf{Optimal}} & \multicolumn{2}{c|}{\textbf{Greedy}} & \multicolumn{2}{c|}{\textbf{SA}} & \multicolumn{2}{c}{\textbf{QAOA-v2}} \\
(N$\times$M) & Cost & Qual. & Cost & Qual. & Cost & Qual. & Cost & Qual. \\
\hline""")

    for r in scaling_results:
        n = r['n_tasks']
        opt = f"${r['optimal_cost']:.2f}" if r.get('optimal_cost') is not None else "---"
        opt_q = f"{r.get('optimal_quality', 0)*100:.0f}\\%" if r.get('optimal_cost') is not None else "---"
        print(f"${n}\\times 3$ & {opt} & {opt_q} & "
              f"${r['greedy_cost']:.2f} & {r['greedy_quality']*100:.0f}\\% & "
              f"${r['sa_cost']:.2f} & {r['sa_quality']*100:.0f}\\% & "
              f"${r['qaoa_cost']:.2f} & {r['qaoa_quality']*100:.0f}\\% \\\\")

    print(r"""\hline
\end{tabular}
\end{table}""")


# ============================================================================
# 4. MAIN
# ============================================================================

if __name__ == '__main__':
    output_dir = "../results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Quantum-Enhanced LLM Routing — Full Benchmark Suite")
    print(f"Cirq {cirq.__version__} | NumPy {np.__version__}")
    print("=" * 70)

    all_results = {}

    # A: Penalty sensitivity
    all_results['penalty'] = experiment_penalty_tuning()
    plot_penalty_sensitivity(all_results['penalty'], output_dir)

    # B: QAOA depth
    all_results['depth'] = experiment_qaoa_depth()
    plot_depth_analysis(all_results['depth'], output_dir)

    # C: Scaling
    all_results['scaling'] = experiment_scaling()
    plot_scaling(all_results['scaling'], output_dir)

    # D: Pareto frontier
    all_results['pareto'] = experiment_pareto()
    plot_pareto(all_results['pareto'], output_dir)

    # Circuit diagram
    plot_circuit_diagram(output_dir)

    # LaTeX table
    generate_latex_table(all_results['scaling'])

    # Save all results
    with open(f"{output_dir}/benchmark_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {output_dir}/benchmark_results.json")

    print("\n" + "=" * 70)
    print("BENCHMARK SUITE COMPLETE")
    print(f"All figures saved to {output_dir}/")
    print("=" * 70)
