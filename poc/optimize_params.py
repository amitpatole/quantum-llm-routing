#!/usr/bin/env python3
"""
Pre-optimize QAOA parameters on simulator before hardware submission.
Finds the best (gamma, beta) for each problem size and depth.
"""

import json
import time
from pathlib import Path

import cirq
import numpy as np
import scipy.optimize

from llm_routing_qaoa import (
    MODELS, generate_task_batch,
    build_cost_matrix, build_quality_penalty, build_latency_penalty,
)
from benchmark_suite import qubo_objective_v2


def build_qaoa_circuit(tasks, models, budget, p, gammas, betas):
    """Build QAOA circuit with given parameters."""
    n, m = len(tasks), len(models)
    num_qubits = n * m
    qubits = cirq.LineQubit.range(num_qubits)

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

    circuit = cirq.Circuit([cirq.H(q) for q in qubits])
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


def evaluate_bitstring(bits, tasks, models):
    """Evaluate a single bitstring: cost, quality, validity."""
    n, m = len(tasks), len(models)
    X = np.array(bits).reshape(n, m)

    valid = all(np.sum(X[i]) == 1 for i in range(n))
    cost = 0.0
    quality_met = 0

    for i, task in enumerate(tasks):
        assigned = np.where(X[i] == 1)[0]
        if len(assigned) == 1:
            j = assigned[0]
            model = models[j]
            cost += model.cost_per_1k_tokens * task.estimated_tokens / 1000
            if model.quality_score >= task.min_quality:
                quality_met += 1

    return cost, quality_met / max(1, n), valid


def optimize_for_config(n_tasks, n_models, p, budget, n_shots=2000,
                        n_restarts=8, max_iter=100):
    """Find optimal (gamma, beta) parameters for a given problem configuration."""
    tasks = generate_task_batch(n_tasks, seed=42)
    models = MODELS[:n_models]
    simulator = cirq.Simulator()

    best_params = None
    best_obj = float('inf')
    best_valid_rate = 0
    best_quality = 0

    def objective(params):
        gammas, betas = list(params[:p]), list(params[p:])
        circ = build_qaoa_circuit(tasks, models, budget, p, gammas, betas)
        res = simulator.run(circ, repetitions=n_shots)
        meas = res.measurements['result']

        total_obj = 0
        for m_bits in meas:
            total_obj += qubo_objective_v2(m_bits, tasks, models, budget)
        return total_obj / len(meas)

    for restart in range(n_restarts):
        init = np.random.uniform(0, np.pi, 2 * p)
        try:
            opt = scipy.optimize.minimize(
                objective, init, method='COBYLA',
                options={'maxiter': max_iter, 'rhobeg': 0.5}
            )
            if opt.fun < best_obj:
                best_obj = opt.fun
                best_params = opt.x.copy()
        except Exception as e:
            print(f"    Restart {restart+1} failed: {e}")
            continue

    # Evaluate best params thoroughly
    if best_params is not None:
        gammas, betas = list(best_params[:p]), list(best_params[p:])
        circ = build_qaoa_circuit(tasks, models, budget, p, gammas, betas)
        res = simulator.run(circ, repetitions=5000)
        meas = res.measurements['result']

        valid_count = 0
        quality_sum = 0
        best_valid_cost = float('inf')
        best_valid_quality = 0

        for m_bits in meas:
            cost, quality, valid = evaluate_bitstring(m_bits, tasks, models)
            if valid:
                valid_count += 1
                quality_sum += quality
                if cost < best_valid_cost:
                    best_valid_cost = cost
                    best_valid_quality = quality

        best_valid_rate = valid_count / len(meas)
        best_quality = quality_sum / max(1, valid_count)

        return {
            'gammas': gammas,
            'betas': betas,
            'qubo_obj': float(best_obj),
            'valid_rate': best_valid_rate,
            'avg_quality': best_quality,
            'best_valid_cost': best_valid_cost if best_valid_cost < float('inf') else None,
            'best_valid_quality': best_valid_quality,
        }

    return None


if __name__ == '__main__':
    print("=" * 70)
    print("QAOA Parameter Pre-Optimization (Simulator)")
    print("Finding best (gamma, beta) for each problem config")
    print("=" * 70)

    configs = [
        # (n_tasks, n_models, p, budget, label)
        (2, 3, 1, 5.0,   "2x3_p1"),
        (2, 3, 2, 5.0,   "2x3_p2"),
        (4, 3, 1, 10.0,  "4x3_p1"),
        (4, 3, 2, 10.0,  "4x3_p2"),
        (4, 3, 3, 10.0,  "4x3_p3"),
        (6, 3, 1, 15.0,  "6x3_p1"),
        (6, 3, 2, 15.0,  "6x3_p2"),
    ]

    results = {}
    for n_tasks, n_models, p, budget, label in configs:
        print(f"\n--- {label}: {n_tasks} tasks x {n_models} models, p={p}, budget=${budget} ---")
        t0 = time.time()
        opt = optimize_for_config(n_tasks, n_models, p, budget,
                                  n_shots=2000, n_restarts=8, max_iter=80)
        elapsed = time.time() - t0

        if opt:
            results[label] = opt
            print(f"  Gammas: {[f'{g:.4f}' for g in opt['gammas']]}")
            print(f"  Betas:  {[f'{b:.4f}' for b in opt['betas']]}")
            print(f"  Valid rate: {opt['valid_rate']:.1%}")
            print(f"  Avg quality (valid): {opt['avg_quality']:.0%}")
            if opt['best_valid_cost'] is not None:
                print(f"  Best valid cost: ${opt['best_valid_cost']:.4f} "
                      f"(quality: {opt['best_valid_quality']:.0%})")
            print(f"  Time: {elapsed:.0f}s")
        else:
            print(f"  FAILED to optimize")

    # Save optimized parameters
    out_path = Path("../results/optimized_params.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"OPTIMIZED PARAMETER SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':>10} | {'Valid %':>8} | {'Avg Qual':>8} | {'Best Cost':>10} | {'Best Qual':>9}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}")
    for label, r in results.items():
        bc = f"${r['best_valid_cost']:.4f}" if r['best_valid_cost'] is not None else "N/A"
        bq = f"{r['best_valid_quality']:.0%}" if r['best_valid_cost'] is not None else "N/A"
        print(f"{label:>10} | {r['valid_rate']:>7.1%} | {r['avg_quality']:>7.0%} | {bc:>10} | {bq:>9}")
