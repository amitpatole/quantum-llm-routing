#!/usr/bin/env python3
"""
Validation Experiment Suite — Paper v2 Additions
==================================================

Runs targeted experiments to validate and extend the findings from v1:

  D. p=1 Scaling:    6, 12, 18 qubits across all 3 backends (validates shallow circuit advantage)
  E. Warm-Start:     Classical-initialized QAOA at p=1 (validates Future Work item 3)
  F. Feasibility Decoding: Post-process all measurements to nearest valid assignment (no QPU)

QPU Budget: 144 seconds (2 min 24 sec)
Estimated Usage: ~12 jobs × ~10s = ~120s

Prerequisites:
    python optimize_params.py       (generates optimized_params.json)
    python ibm_hardware_run.py --setup --token YOUR_KEY

Usage:
    python validation_experiment_suite.py

Author: Amit Patole
"""

import json
from datetime import datetime
from pathlib import Path

import cirq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService

from llm_routing_qaoa import (
    MODELS, generate_task_batch,
    build_cost_matrix, build_quality_penalty, build_latency_penalty,
    solve_greedy,
)
from hardware_experiment_suite import (
    build_qaoa_circuit_cirq,
    cirq_to_qiskit,
    submit_to_backend,
    analyze_measurements,
    run_on_simulator,
)


# ============================================================================
# CONSTANTS
# ============================================================================

QPU_BUDGET_SECONDS = 555.0   # 9m15s remaining (10min total - 45s used in v1+first v2 run)
QPU_WARNING_THRESHOLD = 500.0
N_SHOTS = 4000
BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
OUTPUT_DIR = "../results"


# ============================================================================
# QPU BUDGET TRACKER
# ============================================================================

class QPUBudgetTracker:
    """Tracks cumulative QPU time and enforces budget ceiling."""

    def __init__(self, budget_seconds: float):
        self.budget = budget_seconds
        self.used = 0.0
        self.jobs = []

    def record(self, job_label: str, wait_time: float, estimated_qpu: float = 3.0):
        """Record a job. Uses estimated_qpu (default 3s based on IBM dashboard data)
        instead of wait_time (which includes queue wait, often 60-200s)."""
        self.used += estimated_qpu
        self.jobs.append({'label': job_label, 'wall_time': wait_time, 'est_qpu': estimated_qpu})
        remaining = self.budget - self.used
        print(f"    QPU budget: ~{self.used:.0f}s used, ~{remaining:.0f}s remaining "
              f"(wall: {wait_time:.1f}s, est QPU: {estimated_qpu:.0f}s)")
        if self.used >= QPU_WARNING_THRESHOLD:
            print(f"    *** BUDGET WARNING: {self.used:.0f}s / {self.budget:.0f}s ***")

    def can_submit(self, estimated_time: float = 15.0) -> bool:
        return (self.used + estimated_time) <= self.budget

    def summary(self) -> dict:
        return {
            'total_qpu_seconds': self.used,
            'budget_seconds': self.budget,
            'utilization_pct': round(self.used / self.budget * 100, 1),
            'n_jobs': len(self.jobs),
            'jobs': self.jobs,
        }


# ============================================================================
# WARM-START CIRCUIT
# ============================================================================

def greedy_to_bitstring(greedy_result, tasks, models):
    """Convert greedy assignment dict to binary target vector."""
    n, m = len(tasks), len(models)
    model_name_to_idx = {model.name: j for j, model in enumerate(models)}
    target = np.zeros(n * m, dtype=int)
    for i, task in enumerate(tasks):
        model_name = greedy_result.assignments.get(task.task_id)
        if model_name and model_name in model_name_to_idx:
            j = model_name_to_idx[model_name]
            target[i * m + j] = 1
    return target


def build_warmstart_qaoa_circuit(tasks, models, budget, p, gammas, betas, greedy_result):
    """Build QAOA circuit with warm-start initialization from classical solution.

    Instead of uniform |+>^n superposition (H gates), initializes qubits
    corresponding to the greedy assignment to |1> via X gates, then applies
    H gates to create a superposition biased toward the classical solution.

    The quantum state effect: X then H gives |-> = (|0> - |1>)/sqrt(2),
    while H alone gives |+> = (|0> + |1>)/sqrt(2). The phase difference
    biases QAOA sampling toward solutions near the classical optimum.

    Reference: Egger et al., "Warm-starting Quantum Optimization", Quantum 5, 479 (2021)
    """
    n, m = len(tasks), len(models)
    num_qubits = n * m
    qubits = cirq.LineQubit.range(num_qubits)

    # Get classical solution as binary vector
    target = greedy_to_bitstring(greedy_result, tasks, models)

    # Compute Hamiltonian coefficients (same as cold-start)
    C = build_cost_matrix(tasks, models)
    P = build_quality_penalty(tasks, models)
    L = build_latency_penalty(tasks, models)
    max_cost = np.max(C) if np.max(C) > 0 else 1.0

    h = np.zeros(num_qubits)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            h[idx] = C[i, j] / max_cost + 40.0 * P[i, j] + 3.0 * L[i, j]

    couplings = []
    for i in range(n):
        for j in range(m):
            for k in range(j + 1, m):
                couplings.append((i * m + j, i * m + k, 50.0))

    # WARM-START INITIALIZATION
    init_ops = []
    for idx in range(num_qubits):
        if target[idx] == 1:
            init_ops.append(cirq.X(qubits[idx]))
    init_ops.extend([cirq.H(q) for q in qubits])
    circuit = cirq.Circuit(init_ops)

    # Standard QAOA layers (identical to cold-start)
    for l in range(p):
        ops = []
        for idx in range(num_qubits):
            if abs(h[idx]) > 1e-10:
                ops.append(cirq.rz(2 * gammas[l] * h[idx])(qubits[idx]))
        for a, b, J in couplings:
            ops.append(cirq.ZZPowGate(exponent=2 * gammas[l] * J / np.pi)(
                qubits[a], qubits[b]))
        circuit += cirq.Circuit(ops)
        circuit += cirq.Circuit([cirq.rx(2 * betas[l])(q) for q in qubits])

    circuit += cirq.Circuit([cirq.measure(*qubits, key='result')])
    return circuit


# ============================================================================
# FEASIBILITY DECODING
# ============================================================================

def decode_to_feasible(measurements, tasks, models):
    """Decode each bitstring to the nearest valid assignment.

    For each task i (row of the N x M matrix):
      - If exactly 1 bit set: valid, keep it
      - If 0 bits set: assign to cheapest model meeting quality constraint
      - If 2+ bits set: keep highest-quality model meeting constraints
    """
    n, m = len(tasks), len(models)
    decoded = np.copy(measurements)
    C = build_cost_matrix(tasks, models)

    for shot_idx in range(len(decoded)):
        X = decoded[shot_idx].reshape(n, m)
        for i in range(n):
            row_sum = int(np.sum(X[i]))
            if row_sum == 1:
                continue
            elif row_sum == 0:
                # No assignment: pick cheapest model meeting quality
                best_j = None
                best_cost = float('inf')
                for j, model in enumerate(models):
                    if model.quality_score >= tasks[i].min_quality:
                        if C[i, j] < best_cost:
                            best_cost = C[i, j]
                            best_j = j
                if best_j is None:
                    best_j = int(np.argmin(C[i]))
                X[i] = 0
                X[i, best_j] = 1
            else:
                # Multiple assignments: keep best quality meeting constraints
                set_models = np.where(X[i] == 1)[0]
                chosen_j = None
                best_quality = -1
                for j in set_models:
                    if models[j].quality_score >= tasks[i].min_quality:
                        if models[j].quality_score > best_quality:
                            best_quality = models[j].quality_score
                            chosen_j = j
                if chosen_j is None:
                    chosen_j = min(set_models, key=lambda j: C[i, j])
                X[i] = 0
                X[i, chosen_j] = 1
        decoded[shot_idx] = X.flatten()

    return decoded


# ============================================================================
# EXPERIMENT D: p=1 SCALING ACROSS ALL BACKENDS
# ============================================================================

def run_experiment_d(service, optimized_params, tracker):
    """Run p=1 QAOA at 6/12/18 qubits across all 3 backends."""
    print("\n" + "=" * 70)
    print("EXPERIMENT D: p=1 SCALING ACROSS ALL BACKENDS")
    print("=" * 70)

    scaling_configs = [
        (2, 3, 1, 5.0,  "2x3_p1"),   # 6 qubits
        (4, 3, 1, 10.0, "4x3_p1"),   # 12 qubits
        (6, 3, 1, 15.0, "6x3_p1"),   # 18 qubits
    ]

    results = {}
    raw_measurements = {}

    for n_tasks, n_models, p, budget, param_key in scaling_configs:
        tasks = generate_task_batch(n_tasks, seed=42)
        models = MODELS[:n_models]
        qubits = n_tasks * n_models

        print(f"\n  --- {n_tasks}x{n_models} = {qubits} qubits, p={p} ---")

        # Load optimized params
        params = optimized_params.get(param_key, {})
        gammas = params.get('gammas', [0.5])
        betas = params.get('betas', [0.3])
        print(f"  Params: gamma={[f'{g:.4f}' for g in gammas]}, "
              f"beta={[f'{b:.4f}' for b in betas]}")

        # Build circuit
        cirq_circuit = build_qaoa_circuit_cirq(tasks, models, budget, p, gammas, betas)
        qiskit_circuit, _ = cirq_to_qiskit(cirq_circuit)

        # Simulator baseline
        sim_result = run_on_simulator(cirq_circuit, tasks, models, N_SHOTS)
        print(f"  Simulator: valid={sim_result['valid_rate']:.1%}, "
              f"best_cost=${sim_result['best_valid_cost'] or 'N/A'}")

        # Classical baseline
        greedy = solve_greedy(tasks, models, budget)

        exp_key = f"p1_scaling_{qubits}q"
        results[exp_key] = {
            'config': {'n_tasks': n_tasks, 'n_models': n_models, 'p': p,
                       'qubits': qubits, 'budget': budget},
            'params': {'gammas': gammas, 'betas': betas},
            'classical': {
                'greedy': {'cost': greedy.total_cost, 'quality': greedy.quality_satisfied},
            },
            'simulator': sim_result,
            'hardware': {},
        }

        # Run on each backend
        for backend_name in BACKENDS:
            if not tracker.can_submit():
                print(f"\n  *** BUDGET EXCEEDED — skipping {backend_name} ***")
                continue

            print(f"\n  Backend: {backend_name}")
            try:
                hw = submit_to_backend(qiskit_circuit, service, backend_name, N_SHOTS)
                analysis = analyze_measurements(hw['measurements'], tasks, models)
                analysis['transpiled_depth'] = hw['transpiled_depth']
                analysis['gate_counts'] = hw['gate_counts']
                analysis['job_id'] = hw['job_id']
                tracker.record(f"D_{qubits}q_{backend_name}", hw['wait_time'])

                results[exp_key]['hardware'][backend_name] = analysis

                # Store raw measurements for feasibility decoding
                raw_key = f"D_{qubits}q_{backend_name}"
                raw_measurements[raw_key] = {
                    'measurements': hw['measurements'],
                    'tasks': tasks,
                    'models': models,
                }

                print(f"    Valid: {analysis['valid_rate']:.1%} | "
                      f"Best cost: ${analysis['best_valid_cost'] or 'N/A'} | "
                      f"Quality: {analysis['best_valid_quality']:.0%} | "
                      f"Depth: {hw['transpiled_depth']}")
            except Exception as e:
                print(f"    ERROR: {e}")
                results[exp_key]['hardware'][backend_name] = {'error': str(e)}

    return results, raw_measurements


# ============================================================================
# EXPERIMENT E: WARM-START QAOA
# ============================================================================

def run_experiment_e(service, optimized_params, tracker):
    """Run warm-start QAOA at 6/12/18 qubits on ibm_fez."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E: WARM-START QAOA (Classical Initialization)")
    print("=" * 70)

    configs = [
        (2, 3, 1, 5.0,  "2x3_p1"),   # 6 qubits
        (4, 3, 1, 10.0, "4x3_p1"),   # 12 qubits
        (6, 3, 1, 15.0, "6x3_p1"),   # 18 qubits
    ]

    target_backend = "ibm_fez"
    results = {}
    raw_measurements = {}

    for n_tasks, n_models, p, budget, param_key in configs:
        tasks = generate_task_batch(n_tasks, seed=42)
        models = MODELS[:n_models]
        qubits = n_tasks * n_models

        print(f"\n  --- Warm-start {n_tasks}x{n_models} = {qubits} qubits ---")

        # Get greedy solution for warm-start
        greedy = solve_greedy(tasks, models, budget)
        target_bits = greedy_to_bitstring(greedy, tasks, models)
        print(f"  Greedy solution: cost=${greedy.total_cost:.2f}, "
              f"quality={greedy.quality_satisfied:.0%}")
        print(f"  Warm-start bits: {target_bits.tolist()}")

        # Load optimized params (same as cold-start for fair comparison)
        params = optimized_params.get(param_key, {})
        gammas = params.get('gammas', [0.5])
        betas = params.get('betas', [0.3])

        # Build warm-start circuit
        cirq_circuit = build_warmstart_qaoa_circuit(
            tasks, models, budget, p, gammas, betas, greedy)
        qiskit_circuit, _ = cirq_to_qiskit(cirq_circuit)

        # Simulator baseline
        sim_result = run_on_simulator(cirq_circuit, tasks, models, N_SHOTS)
        print(f"  Simulator: valid={sim_result['valid_rate']:.1%}, "
              f"best_cost=${sim_result['best_valid_cost'] or 'N/A'}")

        exp_key = f"warmstart_{qubits}q"
        results[exp_key] = {
            'config': {'n_tasks': n_tasks, 'n_models': n_models, 'p': p,
                       'qubits': qubits, 'budget': budget, 'init': 'warm-start'},
            'params': {'gammas': gammas, 'betas': betas},
            'greedy_solution': {
                'assignments': greedy.assignments,
                'cost': greedy.total_cost,
                'quality': greedy.quality_satisfied,
                'target_bits': target_bits.tolist(),
            },
            'simulator': sim_result,
            'hardware': {},
        }

        if not tracker.can_submit():
            print(f"\n  *** BUDGET EXCEEDED — skipping hardware ***")
            continue

        print(f"\n  Backend: {target_backend}")
        try:
            hw = submit_to_backend(qiskit_circuit, service, target_backend, N_SHOTS)
            analysis = analyze_measurements(hw['measurements'], tasks, models)
            analysis['transpiled_depth'] = hw['transpiled_depth']
            analysis['gate_counts'] = hw['gate_counts']
            analysis['job_id'] = hw['job_id']
            tracker.record(f"E_{qubits}q_warmstart", hw['wait_time'])

            results[exp_key]['hardware'][target_backend] = analysis

            raw_key = f"E_{qubits}q_warmstart"
            raw_measurements[raw_key] = {
                'measurements': hw['measurements'],
                'tasks': tasks,
                'models': models,
            }

            print(f"    Valid: {analysis['valid_rate']:.1%} | "
                  f"Best cost: ${analysis['best_valid_cost'] or 'N/A'} | "
                  f"Quality: {analysis['best_valid_quality']:.0%} | "
                  f"Depth: {hw['transpiled_depth']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[exp_key]['hardware'][target_backend] = {'error': str(e)}

    return results, raw_measurements


# ============================================================================
# FEASIBILITY DECODING (NO QPU)
# ============================================================================

def run_feasibility_decoding(raw_measurements_store):
    """Apply feasibility decoding to all raw measurements."""
    print("\n" + "=" * 70)
    print("FEASIBILITY DECODING (Post-Processing — No QPU)")
    print("=" * 70)

    decoding_results = {}

    for label, data in raw_measurements_store.items():
        measurements = data['measurements']
        tasks = data['tasks']
        models = data['models']

        # Original analysis
        original = analyze_measurements(measurements, tasks, models)

        # Decode to feasible
        decoded_meas = decode_to_feasible(measurements, tasks, models)
        decoded = analyze_measurements(decoded_meas, tasks, models)

        improvement = decoded['valid_rate'] - original['valid_rate']

        decoding_results[label] = {
            'original_valid_rate': original['valid_rate'],
            'decoded_valid_rate': decoded['valid_rate'],
            'valid_rate_improvement': improvement,
            'original_best_cost': original['best_valid_cost'],
            'decoded_best_cost': decoded['best_valid_cost'],
            'original_mean_cost': original['mean_cost'],
            'decoded_mean_cost': decoded['mean_cost'],
            'original_quality': original['best_valid_quality'],
            'decoded_quality': decoded['best_valid_quality'],
            'decoded_mean_quality': decoded['mean_quality'],
        }

        print(f"\n  {label}:")
        print(f"    Raw valid:     {original['valid_rate']:>7.1%}  →  "
              f"Decoded: {decoded['valid_rate']:>7.1%}")
        print(f"    Raw best cost: ${original['best_valid_cost'] or 0:.2f}  →  "
              f"Decoded: ${decoded['best_valid_cost'] or 0:.2f}")
        print(f"    Decoded mean quality: {decoded['mean_quality']:.1%}")

    return decoding_results


# ============================================================================
# PUBLICATION FIGURES
# ============================================================================

def plot_fig14_p1_scaling(exp_d_results, old_results, output_dir):
    """Figure 14: p=1 valid rates across all backends, overlaid with p=2 from v1."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Collect p=1 data from Experiment D
    scaling_keys = sorted(
        [k for k in exp_d_results if k.startswith('p1_scaling_')],
        key=lambda k: exp_d_results[k]['config']['qubits']
    )

    if not scaling_keys:
        plt.close()
        return

    qubits_list = [exp_d_results[k]['config']['qubits'] for k in scaling_keys]
    sim_valid_p1 = [exp_d_results[k]['simulator']['valid_rate'] * 100 for k in scaling_keys]

    backend_colors = {'ibm_fez': '#E91E63', 'ibm_kingston': '#00BCD4', 'ibm_marrakesh': '#FF9800'}

    # (a) Valid rate: p=1 vs p=2
    ax = axes[0]
    ax.plot(qubits_list, sim_valid_p1, 'o-', color='#9C27B0', linewidth=2,
            markersize=8, label='p=1 Simulator', zorder=5)

    for bn in BACKENDS:
        hw_valid = []
        for k in scaling_keys:
            hw = exp_d_results[k].get('hardware', {}).get(bn, {})
            hw_valid.append(hw.get('valid_rate', 0) * 100)
        ax.plot(qubits_list, hw_valid, 's-', color=backend_colors[bn],
                linewidth=2, markersize=7, label=f'p=1 {bn.replace("ibm_", "")}')

    # Overlay p=2 data from v1 (dashed)
    old_exps = old_results.get('experiments', {})
    p2_keys = ['scaling_6q', 'scaling_12q', 'scaling_18q']
    p2_qubits = [6, 12, 18]
    for bn in BACKENDS:
        p2_valid = []
        for pk in p2_keys:
            hw = old_exps.get(pk, {}).get('hardware', {}).get(bn, {})
            p2_valid.append(hw.get('valid_rate', 0) * 100)
        ax.plot(p2_qubits, p2_valid, 'x--', color=backend_colors[bn],
                linewidth=1, markersize=6, alpha=0.5,
                label=f'p=2 {bn.replace("ibm_", "")}')

    ax.set_xlabel('Qubits (N x M)', fontsize=11)
    ax.set_ylabel('Valid Assignments (%)', fontsize=11)
    ax.set_title('(a) p=1 vs p=2 Valid Rate')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # (b) Best valid cost
    ax = axes[1]
    for bn in BACKENDS:
        hw_costs = []
        for k in scaling_keys:
            hw = exp_d_results[k].get('hardware', {}).get(bn, {})
            c = hw.get('best_valid_cost')
            hw_costs.append(c if c is not None else 0)
        ax.plot(qubits_list, hw_costs, 's-', color=backend_colors[bn],
                linewidth=2, markersize=7, label=f'p=1 {bn.replace("ibm_", "")}')

    # Greedy reference
    for k in scaling_keys:
        classical = exp_d_results[k].get('classical', {})
        if classical:
            q = exp_d_results[k]['config']['qubits']
            ax.scatter([q], [classical['greedy']['cost']], marker='D',
                       color='green', s=60, zorder=10,
                       label='Greedy' if k == scaling_keys[0] else '')

    ax.set_xlabel('Qubits (N x M)', fontsize=11)
    ax.set_ylabel('Best Valid Cost (USD)', fontsize=11)
    ax.set_title('(b) Best Solution Cost')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Transpiled depth (p=1 should be ~52 everywhere)
    ax = axes[2]
    for bn in BACKENDS:
        depths = []
        for k in scaling_keys:
            hw = exp_d_results[k].get('hardware', {}).get(bn, {})
            depths.append(hw.get('transpiled_depth', 0))
        ax.plot(qubits_list, depths, 's-', color=backend_colors[bn],
                linewidth=2, markersize=7, label=f'p=1 {bn.replace("ibm_", "")}')

    # Overlay p=2 depths
    for bn in BACKENDS:
        p2_depths = []
        for pk in p2_keys:
            hw = old_exps.get(pk, {}).get('hardware', {}).get(bn, {})
            p2_depths.append(hw.get('transpiled_depth', 0))
        ax.plot(p2_qubits, p2_depths, 'x--', color=backend_colors[bn],
                linewidth=1, markersize=6, alpha=0.5,
                label=f'p=2 {bn.replace("ibm_", "")}')

    ax.set_xlabel('Qubits (N x M)', fontsize=11)
    ax.set_ylabel('Transpiled Circuit Depth', fontsize=11)
    ax.set_title('(c) Circuit Depth: p=1 vs p=2')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.suptitle('Shallow Circuit Advantage: p=1 Scaling on IBM Quantum Hardware',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig14_p1_scaling.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig14_p1_scaling.png")
    plt.close()


def plot_fig15_warmstart(exp_d_results, exp_e_results, output_dir):
    """Figure 15: Warm-start vs cold-start comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    qubit_sizes = [6, 12, 18]
    cold_valid = []
    warm_valid = []
    cold_costs = []
    warm_costs = []

    for q in qubit_sizes:
        # Cold-start from Experiment D (ibm_fez)
        cold_key = f"p1_scaling_{q}q"
        cold_hw = exp_d_results.get(cold_key, {}).get('hardware', {}).get('ibm_fez', {})
        cold_valid.append(cold_hw.get('valid_rate', 0) * 100)
        cold_costs.append(cold_hw.get('best_valid_cost') or 0)

        # Warm-start from Experiment E (ibm_fez)
        warm_key = f"warmstart_{q}q"
        warm_hw = exp_e_results.get(warm_key, {}).get('hardware', {}).get('ibm_fez', {})
        warm_valid.append(warm_hw.get('valid_rate', 0) * 100)
        warm_costs.append(warm_hw.get('best_valid_cost') or 0)

    x = np.arange(len(qubit_sizes))
    width = 0.35

    # (a) Valid rate comparison
    ax1.bar(x - width/2, cold_valid, width, label='Cold-Start (|+>^n)',
            color='#9C27B0', alpha=0.8)
    ax1.bar(x + width/2, warm_valid, width, label='Warm-Start (Greedy)',
            color='#4CAF50', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{q}q' for q in qubit_sizes])
    ax1.set_ylabel('Valid Assignments (%)', fontsize=11)
    ax1.set_title('(a) Valid Rate: Cold vs Warm Start')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis='y')
    for i, (cv, wv) in enumerate(zip(cold_valid, warm_valid)):
        if cv > 0:
            ax1.text(i - width/2, cv + 0.5, f'{cv:.1f}%', ha='center', fontsize=8)
        if wv > 0:
            ax1.text(i + width/2, wv + 0.5, f'{wv:.1f}%', ha='center', fontsize=8)

    # (b) Best cost comparison
    ax2.bar(x - width/2, cold_costs, width, label='Cold-Start',
            color='#9C27B0', alpha=0.8)
    ax2.bar(x + width/2, warm_costs, width, label='Warm-Start',
            color='#4CAF50', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{q}q' for q in qubit_sizes])
    ax2.set_ylabel('Best Valid Cost (USD)', fontsize=11)
    ax2.set_title('(b) Best Cost: Cold vs Warm Start')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')

    plt.suptitle('Warm-Start QAOA: Classical Initialization vs Standard (p=1, ibm_fez)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig15_warmstart_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig15_warmstart_comparison.png")
    plt.close()


def plot_fig16_decoding(decoding_results, output_dir):
    """Figure 16: Raw vs decoded valid rates and costs."""
    if not decoding_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(decoding_results.keys())
    short_labels = [l.replace('D_', '').replace('E_', 'W').replace('ibm_', '')
                    .replace('_warmstart', ' WS') for l in labels]
    raw_valid = [decoding_results[l]['original_valid_rate'] * 100 for l in labels]
    dec_valid = [decoding_results[l]['decoded_valid_rate'] * 100 for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    # (a) Valid rate improvement
    ax1.bar(x - width/2, raw_valid, width, label='Raw QAOA', color='#E91E63', alpha=0.8)
    ax1.bar(x + width/2, dec_valid, width, label='After Decoding', color='#4CAF50', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=8, rotation=45, ha='right')
    ax1.set_ylabel('Valid Assignments (%)', fontsize=11)
    ax1.set_title('(a) Feasibility Decoding: Valid Rate')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis='y')

    # (b) Cost comparison
    raw_costs = [decoding_results[l]['original_mean_cost'] for l in labels]
    dec_costs = [decoding_results[l]['decoded_mean_cost'] for l in labels]
    dec_quality = [decoding_results[l]['decoded_mean_quality'] * 100 for l in labels]

    ax2.bar(x - width/2, raw_costs, width, label='Raw Mean Cost', color='#E91E63', alpha=0.8)
    ax2.bar(x + width/2, dec_costs, width, label='Decoded Mean Cost', color='#4CAF50', alpha=0.8)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, dec_quality, 'ko-', markersize=5, label='Decoded Quality %')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=8, rotation=45, ha='right')
    ax2.set_ylabel('Mean Cost (USD)', fontsize=11)
    ax2_twin.set_ylabel('Quality Satisfaction (%)', fontsize=11)
    ax2_twin.set_ylim(0, 110)
    ax2.set_title('(b) Decoded Cost & Quality')
    ax2.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3, axis='y')

    plt.suptitle('Feasibility-First Decoding: Projecting Invalid Measurements to Valid Assignments',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig16_feasibility_decoding.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig16_feasibility_decoding.png")
    plt.close()


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_validation_summary(exp_d_results, exp_e_results, decoding_results, tracker):
    """Print publication-ready summary tables."""

    print("\n" + "=" * 90)
    print("VALIDATION RESULTS — PUBLICATION SUMMARY")
    print("=" * 90)

    # --- Experiment D ---
    print("\n--- EXPERIMENT D: p=1 SCALING ---")
    print(f"{'Size':>8} | {'Backend':>12} | {'Valid %':>8} | {'Best Cost':>10} | "
          f"{'Quality':>8} | {'Depth':>6} | {'Job ID':>24}")
    print("-" * 90)

    for exp_key in sorted(exp_d_results.keys()):
        exp = exp_d_results[exp_key]
        label = f"{exp['config']['qubits']}q p=1"

        sim = exp.get('simulator', {})
        bc = f"${sim.get('best_valid_cost', 0):.2f}" if sim.get('best_valid_cost') else "N/A"
        print(f"{label:>8} | {'Simulator':>12} | {sim.get('valid_rate', 0):>7.1%} | "
              f"{bc:>10} | {sim.get('best_valid_quality', 0):>7.0%} | {'---':>6} | {'---':>24}")

        for bn, hw in exp.get('hardware', {}).items():
            if isinstance(hw, dict) and 'error' not in hw:
                bc = f"${hw.get('best_valid_cost', 0):.2f}" if hw.get('best_valid_cost') else "N/A"
                short = bn.replace('ibm_', '')
                job_id = hw.get('job_id', 'N/A')
                print(f"{'':>8} | {short:>12} | {hw.get('valid_rate', 0):>7.1%} | "
                      f"{bc:>10} | {hw.get('best_valid_quality', 0):>7.0%} | "
                      f"{hw.get('transpiled_depth', 0):>6} | {job_id:>24}")

    # --- Experiment E ---
    print("\n--- EXPERIMENT E: WARM-START vs COLD-START (ibm_fez, p=1) ---")
    print(f"{'Size':>8} | {'Init':>12} | {'Valid %':>8} | {'Best Cost':>10} | "
          f"{'Quality':>8} | {'Depth':>6}")
    print("-" * 70)

    for q in [6, 12, 18]:
        # Cold-start
        cold_key = f"p1_scaling_{q}q"
        cold_hw = exp_d_results.get(cold_key, {}).get('hardware', {}).get('ibm_fez', {})
        if cold_hw and 'error' not in cold_hw:
            bc = f"${cold_hw.get('best_valid_cost', 0):.2f}" if cold_hw.get('best_valid_cost') else "N/A"
            print(f"{q}q p=1{'':>3} | {'Cold |+>^n':>12} | {cold_hw.get('valid_rate', 0):>7.1%} | "
                  f"{bc:>10} | {cold_hw.get('best_valid_quality', 0):>7.0%} | "
                  f"{cold_hw.get('transpiled_depth', 0):>6}")

        # Warm-start
        warm_key = f"warmstart_{q}q"
        warm_hw = exp_e_results.get(warm_key, {}).get('hardware', {}).get('ibm_fez', {})
        if warm_hw and 'error' not in warm_hw:
            bc = f"${warm_hw.get('best_valid_cost', 0):.2f}" if warm_hw.get('best_valid_cost') else "N/A"
            print(f"{'':>8} | {'Warm-Start':>12} | {warm_hw.get('valid_rate', 0):>7.1%} | "
                  f"{bc:>10} | {warm_hw.get('best_valid_quality', 0):>7.0%} | "
                  f"{warm_hw.get('transpiled_depth', 0):>6}")

    # --- Feasibility Decoding ---
    print("\n--- FEASIBILITY DECODING ---")
    print(f"{'Experiment':>24} | {'Raw Valid':>10} | {'Decoded':>10} | "
          f"{'Raw Cost':>10} | {'Dec Cost':>10} | {'Dec Quality':>12}")
    print("-" * 90)

    for label, data in decoding_results.items():
        print(f"{label:>24} | {data['original_valid_rate']:>9.1%} | "
              f"{data['decoded_valid_rate']:>9.1%} | "
              f"${data['original_mean_cost']:>8.2f} | "
              f"${data['decoded_mean_cost']:>8.2f} | "
              f"{data['decoded_mean_quality']:>10.1%}")

    # --- QPU Budget ---
    budget = tracker.summary()
    print(f"\n--- QPU BUDGET ---")
    print(f"Total QPU time: {budget['total_qpu_seconds']:.1f}s "
          f"({budget['total_qpu_seconds']/60:.1f} min)")
    print(f"Budget: {budget['budget_seconds']:.0f}s "
          f"({budget['utilization_pct']:.1f}% utilized)")
    print(f"Jobs submitted: {budget['n_jobs']}")
    for job in budget['jobs']:
        print(f"  {job['label']:>30}: ~{job['est_qpu']:.0f}s QPU (wall: {job['wall_time']:.1f}s)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    output_dir = OUTPUT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VALIDATION EXPERIMENT SUITE (Paper v2)")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"QPU Budget: {QPU_BUDGET_SECONDS:.0f}s ({QPU_BUDGET_SECONDS/60:.1f} min)")
    print("=" * 70)

    # Load optimized parameters
    params_file = Path("../results/optimized_params.json")
    if params_file.exists():
        with open(params_file) as f:
            optimized_params = json.load(f)
        print(f"Loaded optimized params for {len(optimized_params)} configs")
    else:
        print("WARNING: No optimized_params.json found. Using defaults.")
        optimized_params = {}

    # Load old v1 results for comparison figures
    old_file = Path("../results/hardware_full_results.json")
    if old_file.exists():
        with open(old_file) as f:
            old_results = json.load(f)
        print(f"Loaded v1 results ({len(old_results.get('experiments', {}))} experiments)")
    else:
        old_results = {}

    # Connect to IBM Quantum
    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backends_available = [b.name for b in service.backends(operational=True)]
    print(f"Available backends: {backends_available}")

    # Initialize budget tracker
    tracker = QPUBudgetTracker(QPU_BUDGET_SECONDS)

    # Raw measurements store for feasibility decoding
    all_raw_measurements = {}

    # ===== EXPERIMENT D: p=1 SCALING =====
    exp_d_results, exp_d_raw = run_experiment_d(service, optimized_params, tracker)
    all_raw_measurements.update(exp_d_raw)

    # ===== EXPERIMENT E: WARM-START =====
    exp_e_results, exp_e_raw = run_experiment_e(service, optimized_params, tracker)
    all_raw_measurements.update(exp_e_raw)

    # ===== FEASIBILITY DECODING (no QPU) =====
    decoding_results = run_feasibility_decoding(all_raw_measurements)

    # ===== SAVE RESULTS =====
    all_validation = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_shots': N_SHOTS,
            'backends': BACKENDS,
            'qpu_budget': tracker.summary(),
            'paper_version': 'v2',
            'extends': 'hardware_full_results.json (v1)',
        },
        'experiment_d': exp_d_results,
        'experiment_e': exp_e_results,
        'feasibility_decoding': decoding_results,
    }

    # Clean for JSON serialization (remove numpy arrays, large distributions)
    results_clean = json.loads(json.dumps(all_validation, default=str))
    for section in ['experiment_d', 'experiment_e']:
        for exp_key in results_clean.get(section, {}):
            exp = results_clean[section][exp_key]
            if 'simulator' in exp:
                exp['simulator'].pop('cost_distribution', None)
                exp['simulator'].pop('quality_distribution', None)
            for bn in exp.get('hardware', {}):
                if isinstance(exp['hardware'][bn], dict):
                    exp['hardware'][bn].pop('cost_distribution', None)
                    exp['hardware'][bn].pop('quality_distribution', None)

    with open(f"{output_dir}/validation_results.json", 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nSaved: {output_dir}/validation_results.json")

    # ===== GENERATE FIGURES =====
    print("\nGenerating publication figures...")
    plot_fig14_p1_scaling(exp_d_results, old_results, output_dir)
    plot_fig15_warmstart(exp_d_results, exp_e_results, output_dir)
    plot_fig16_decoding(decoding_results, output_dir)

    # ===== PRINT SUMMARY =====
    print_validation_summary(exp_d_results, exp_e_results, decoding_results, tracker)

    print("\n" + "=" * 70)
    print("VALIDATION SUITE COMPLETE")
    print("=" * 70)
