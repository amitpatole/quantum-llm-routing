#!/usr/bin/env python3
"""
Comprehensive IBM Quantum Hardware Experiment Suite
=====================================================

Runs a strategic set of experiments within ~3-4 minutes of QPU time
to produce publication-quality data for the research paper.

Experiments:
  A. Scaling:    6, 12, 18 qubits across all 3 backends
  B. Depth:      p=1, 2, 3 on 12 qubits (best backend)
  C. Optimized:  Best simulator-tuned params, high shot count
  D. Cross-backend: Same circuit on all 3 backends

Prerequisites:
    python optimize_params.py   (generates optimized_params.json)
    python ibm_hardware_run.py --setup --token YOUR_KEY

Usage:
    python hardware_experiment_suite.py

Author: Amit Patole
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cirq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import qasm2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from llm_routing_qaoa import (
    MODELS, generate_task_batch,
    build_cost_matrix, build_quality_penalty, build_latency_penalty,
    solve_greedy, solve_simulated_annealing, solve_brute_force,
)


# ============================================================================
# CIRCUIT BUILDING
# ============================================================================

def build_qaoa_circuit_cirq(tasks, models, budget, p, gammas, betas):
    """Build QAOA circuit in Cirq with given parameters."""
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


def cirq_to_qiskit(cirq_circuit):
    """Convert Cirq circuit to Qiskit via QASM with sx/sxdg support."""
    qasm_str = cirq.qasm(cirq_circuit, args=cirq.QasmArgs(version="2.0"))
    gate_defs = (
        '\n// Custom gate definitions for Cirq compatibility\n'
        'gate sx a { rz(-pi/2) a; ry(pi/2) a; rz(pi/2) a; }\n'
        'gate sxdg a { rz(-pi/2) a; ry(-pi/2) a; rz(pi/2) a; }\n'
    )
    qasm_str = qasm_str.replace(
        'include "qelib1.inc";',
        'include "qelib1.inc";' + gate_defs,
    )
    return qasm2.loads(qasm_str), qasm_str


# ============================================================================
# HARDWARE SUBMISSION
# ============================================================================

def submit_to_backend(qiskit_circuit, service, backend_name, n_shots):
    """Submit circuit to a specific IBM backend and return results."""
    backend = service.backend(backend_name)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qiskit_circuit)

    transpiled_depth = isa_circuit.depth()
    gate_counts = isa_circuit.count_ops()

    sampler = SamplerV2(backend)
    job = sampler.run([isa_circuit], shots=n_shots)
    job_id = job.job_id()

    print(f"    Submitted to {backend_name} (job: {job_id}, "
          f"depth: {transpiled_depth}, shots: {n_shots})")

    t0 = time.time()
    result = job.result()
    wait_time = time.time() - t0

    # Extract bitstrings
    pub_result = result[0]
    bitarray = None
    for attr in ['m_result', 'meas', 'c', 'result']:
        if hasattr(pub_result.data, attr):
            bitarray = getattr(pub_result.data, attr)
            break
    if bitarray is None:
        for attr in dir(pub_result.data):
            if not attr.startswith('_'):
                bitarray = getattr(pub_result.data, attr)
                break

    bitstrings = bitarray.get_bitstrings()
    measurements = np.array([[int(b) for b in bs] for bs in bitstrings])

    print(f"    Received in {wait_time:.1f}s")

    return {
        'measurements': measurements,
        'backend': backend_name,
        'job_id': job_id,
        'transpiled_depth': transpiled_depth,
        'gate_counts': {k: int(v) for k, v in gate_counts.items()},
        'wait_time': wait_time,
    }


def analyze_measurements(measurements, tasks, models):
    """Analyze measurement results."""
    n, m = len(tasks), len(models)
    valid_count = 0
    costs = []
    qualities = []
    best_valid_cost = float('inf')
    best_valid_quality = 0
    best_valid_assignment = None

    for bits in measurements:
        X = np.array(bits).reshape(n, m)
        is_valid = all(np.sum(X[i]) == 1 for i in range(n))

        cost = 0.0
        q_met = 0
        assignment = {}
        for i, task in enumerate(tasks):
            assigned = np.where(X[i] == 1)[0]
            if len(assigned) > 0:
                j = assigned[0]
                model = models[j]
                assignment[task.task_id] = model.name
                cost += model.cost_per_1k_tokens * task.estimated_tokens / 1000
                if model.quality_score >= task.min_quality:
                    q_met += 1
            else:
                assignment[task.task_id] = "UNASSIGNED"

        costs.append(cost)
        quality = q_met / max(1, n)
        qualities.append(quality)

        if is_valid:
            valid_count += 1
            if cost < best_valid_cost:
                best_valid_cost = cost
                best_valid_quality = quality
                best_valid_assignment = assignment

    return {
        'n_shots': len(measurements),
        'valid_count': valid_count,
        'valid_rate': valid_count / len(measurements),
        'mean_cost': float(np.mean(costs)),
        'std_cost': float(np.std(costs)),
        'mean_quality': float(np.mean(qualities)),
        'best_valid_cost': float(best_valid_cost) if best_valid_cost < float('inf') else None,
        'best_valid_quality': best_valid_quality,
        'best_assignment': best_valid_assignment,
        'cost_distribution': [float(c) for c in costs],
        'quality_distribution': [float(q) for q in qualities],
    }


def run_on_simulator(cirq_circuit, tasks, models, n_shots):
    """Run on Cirq simulator and analyze."""
    simulator = cirq.Simulator()
    result = simulator.run(cirq_circuit, repetitions=n_shots)
    measurements = result.measurements['result']
    return analyze_measurements(measurements, tasks, models)


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_all_experiments(service, backends, optimized_params, n_shots=4000):
    """Run the full experiment suite."""

    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_shots': n_shots,
            'backends': backends,
        },
        'experiments': {},
    }

    time_tracker = {'total_qpu_seconds': 0}

    # =====================================================================
    # EXPERIMENT A: SCALING (6, 12, 18 qubits) on all backends
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT A: SCALING ACROSS PROBLEM SIZES")
    print("=" * 70)

    scaling_configs = [
        (2, 3, 2, 5.0,  "2x3_p2"),   # 6 qubits
        (4, 3, 2, 10.0, "4x3_p2"),   # 12 qubits
        (6, 3, 2, 15.0, "6x3_p2"),   # 18 qubits
    ]

    for n_tasks, n_models, p, budget, param_key in scaling_configs:
        tasks = generate_task_batch(n_tasks, seed=42)
        models = MODELS[:n_models]
        qubits = n_tasks * n_models

        print(f"\n  --- {n_tasks}x{n_models} = {qubits} qubits ---")

        # Get optimized params (fallback to defaults)
        params = optimized_params.get(param_key, {})
        gammas = params.get('gammas', [0.5] * p)
        betas = params.get('betas', [0.3] * p)
        print(f"  Params: gamma={[f'{g:.3f}' for g in gammas]}, "
              f"beta={[f'{b:.3f}' for b in betas]}")

        # Build circuit
        cirq_circuit = build_qaoa_circuit_cirq(tasks, models, budget, p, gammas, betas)
        qiskit_circuit, _ = cirq_to_qiskit(cirq_circuit)

        # Simulator baseline
        sim_result = run_on_simulator(cirq_circuit, tasks, models, n_shots)
        print(f"  Simulator: valid={sim_result['valid_rate']:.1%}, "
              f"best_cost=${sim_result['best_valid_cost'] or 'N/A'}")

        # Classical baselines
        greedy = solve_greedy(tasks, models, budget)
        sa = solve_simulated_annealing(tasks, models, budget)

        exp_key = f"scaling_{qubits}q"
        all_results['experiments'][exp_key] = {
            'config': {'n_tasks': n_tasks, 'n_models': n_models, 'p': p,
                       'qubits': qubits, 'budget': budget},
            'params': {'gammas': gammas, 'betas': betas},
            'classical': {
                'greedy': {'cost': greedy.total_cost, 'quality': greedy.quality_satisfied},
                'sa': {'cost': sa.total_cost, 'quality': sa.quality_satisfied},
            },
            'simulator': sim_result,
            'hardware': {},
        }

        # Run on each backend
        for backend_name in backends:
            print(f"\n  Backend: {backend_name}")
            try:
                hw = submit_to_backend(qiskit_circuit, service, backend_name, n_shots)
                analysis = analyze_measurements(hw['measurements'], tasks, models)
                analysis['transpiled_depth'] = hw['transpiled_depth']
                analysis['gate_counts'] = hw['gate_counts']
                analysis['job_id'] = hw['job_id']
                time_tracker['total_qpu_seconds'] += hw['wait_time']

                all_results['experiments'][exp_key]['hardware'][backend_name] = analysis

                print(f"    Valid: {analysis['valid_rate']:.1%} | "
                      f"Best cost: ${analysis['best_valid_cost'] or 'N/A'} | "
                      f"Quality: {analysis['best_valid_quality']:.0%}")
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results['experiments'][exp_key]['hardware'][backend_name] = {'error': str(e)}

        print(f"\n  QPU time used so far: ~{time_tracker['total_qpu_seconds']:.0f}s")

    # =====================================================================
    # EXPERIMENT B: DEPTH ANALYSIS (p=1,2,3 on 12 qubits, best backend)
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT B: CIRCUIT DEPTH ANALYSIS (p=1,2,3)")
    print("=" * 70)

    best_backend = backends[0]  # Use first (usually least busy)
    tasks = generate_task_batch(4, seed=42)
    models_3 = MODELS[:3]

    for p in [1, 2, 3]:
        param_key = f"4x3_p{p}"
        params = optimized_params.get(param_key, {})
        gammas = params.get('gammas', [0.5] * p)
        betas = params.get('betas', [0.3] * p)

        print(f"\n  --- p={p}, params from simulator optimization ---")

        cirq_circuit = build_qaoa_circuit_cirq(tasks, models_3, 10.0, p, gammas, betas)
        qiskit_circuit, _ = cirq_to_qiskit(cirq_circuit)

        # Simulator
        sim_result = run_on_simulator(cirq_circuit, tasks, models_3, n_shots)

        exp_key = f"depth_p{p}"
        all_results['experiments'][exp_key] = {
            'config': {'n_tasks': 4, 'n_models': 3, 'p': p, 'qubits': 12},
            'params': {'gammas': gammas, 'betas': betas},
            'simulator': sim_result,
            'hardware': {},
        }

        try:
            hw = submit_to_backend(qiskit_circuit, service, best_backend, n_shots)
            analysis = analyze_measurements(hw['measurements'], tasks, models_3)
            analysis['transpiled_depth'] = hw['transpiled_depth']
            analysis['gate_counts'] = hw['gate_counts']
            analysis['job_id'] = hw['job_id']
            time_tracker['total_qpu_seconds'] += hw['wait_time']

            all_results['experiments'][exp_key]['hardware'][best_backend] = analysis

            print(f"  Sim:  valid={sim_result['valid_rate']:.1%}")
            print(f"  HW:   valid={analysis['valid_rate']:.1%}, "
                  f"best=${analysis['best_valid_cost'] or 'N/A'}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print(f"  QPU time used so far: ~{time_tracker['total_qpu_seconds']:.0f}s")

    # =====================================================================
    # EXPERIMENT C: HIGH-SHOT OPTIMIZED RUN (best params, 8000 shots)
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT C: HIGH-SHOT OPTIMIZED RUN (8000 shots)")
    print("=" * 70)

    params = optimized_params.get("4x3_p2", {})
    gammas = params.get('gammas', [0.5, 0.5])
    betas = params.get('betas', [0.3, 0.3])

    cirq_circuit = build_qaoa_circuit_cirq(tasks, models_3, 10.0, 2, gammas, betas)
    qiskit_circuit, _ = cirq_to_qiskit(cirq_circuit)

    sim_result = run_on_simulator(cirq_circuit, tasks, models_3, 8000)

    exp_key = "high_shot"
    all_results['experiments'][exp_key] = {
        'config': {'n_tasks': 4, 'n_models': 3, 'p': 2, 'qubits': 12, 'shots': 8000},
        'params': {'gammas': gammas, 'betas': betas},
        'simulator': sim_result,
        'hardware': {},
    }

    try:
        hw = submit_to_backend(qiskit_circuit, service, best_backend, 8000)
        analysis = analyze_measurements(hw['measurements'], tasks, models_3)
        analysis['transpiled_depth'] = hw['transpiled_depth']
        analysis['gate_counts'] = hw['gate_counts']
        analysis['job_id'] = hw['job_id']
        time_tracker['total_qpu_seconds'] += hw['wait_time']

        all_results['experiments'][exp_key]['hardware'][best_backend] = analysis
        print(f"  Sim:  valid={sim_result['valid_rate']:.1%}, "
              f"best=${sim_result['best_valid_cost'] or 'N/A'}")
        print(f"  HW:   valid={analysis['valid_rate']:.1%}, "
              f"best=${analysis['best_valid_cost'] or 'N/A'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    all_results['metadata']['total_qpu_seconds'] = time_tracker['total_qpu_seconds']
    return all_results


# ============================================================================
# PUBLICATION FIGURES
# ============================================================================

def generate_all_figures(results, output_dir):
    """Generate all publication-quality figures from hardware results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    experiments = results['experiments']

    # === Figure 10: Scaling — Valid Rate: Simulator vs Hardware ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scaling_keys = [k for k in experiments if k.startswith('scaling_')]
    scaling_keys.sort(key=lambda k: experiments[k]['config']['qubits'])

    if scaling_keys:
        qubits_list = [experiments[k]['config']['qubits'] for k in scaling_keys]
        sim_valid = [experiments[k]['simulator']['valid_rate'] * 100 for k in scaling_keys]

        # Per-backend valid rates
        backends = results['metadata']['backends']
        backend_colors = ['#E91E63', '#00BCD4', '#FF9800']

        ax = axes[0]
        ax.plot(qubits_list, sim_valid, 'o-', color='#9C27B0', linewidth=2,
                markersize=8, label='Simulator', zorder=5)
        for idx, bn in enumerate(backends):
            hw_valid = []
            for k in scaling_keys:
                hw = experiments[k].get('hardware', {}).get(bn, {})
                hw_valid.append(hw.get('valid_rate', 0) * 100)
            ax.plot(qubits_list, hw_valid, 's--', color=backend_colors[idx],
                    linewidth=1.5, markersize=7, label=bn)
        ax.set_xlabel('Qubits (N x M)', fontsize=11)
        ax.set_ylabel('Valid Assignments (%)', fontsize=11)
        ax.set_title('(a) Constraint Satisfaction vs Size')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Best valid cost
        ax = axes[1]
        sim_costs = []
        for k in scaling_keys:
            c = experiments[k]['simulator'].get('best_valid_cost')
            sim_costs.append(c if c is not None else 0)
        ax.plot(qubits_list, sim_costs, 'o-', color='#9C27B0', linewidth=2,
                markersize=8, label='Simulator')
        for idx, bn in enumerate(backends):
            hw_costs = []
            for k in scaling_keys:
                hw = experiments[k].get('hardware', {}).get(bn, {})
                c = hw.get('best_valid_cost')
                hw_costs.append(c if c is not None else 0)
            ax.plot(qubits_list, hw_costs, 's--', color=backend_colors[idx],
                    linewidth=1.5, markersize=7, label=bn)

        # Add classical reference lines
        for k in scaling_keys:
            classical = experiments[k].get('classical', {})
            if classical:
                q = experiments[k]['config']['qubits']
                ax.scatter([q], [classical['greedy']['cost']], marker='D',
                           color='green', s=60, zorder=10)
        ax.set_xlabel('Qubits (N x M)', fontsize=11)
        ax.set_ylabel('Best Valid Cost (USD)', fontsize=11)
        ax.set_title('(b) Best Solution Cost vs Size')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Transpiled depth
        ax = axes[2]
        for idx, bn in enumerate(backends):
            depths = []
            for k in scaling_keys:
                hw = experiments[k].get('hardware', {}).get(bn, {})
                depths.append(hw.get('transpiled_depth', 0))
            ax.plot(qubits_list, depths, 's-', color=backend_colors[idx],
                    linewidth=2, markersize=7, label=bn)
        ax.set_xlabel('Qubits (N x M)', fontsize=11)
        ax.set_ylabel('Transpiled Circuit Depth', fontsize=11)
        ax.set_title('(c) Circuit Depth After Transpilation')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('QAOA Scaling on IBM Quantum Hardware (Heron 156-qubit)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig10_hw_scaling.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig10_hw_scaling.png")
    plt.close()

    # === Figure 11: Depth Analysis — Sim vs Hardware ===
    depth_keys = [k for k in experiments if k.startswith('depth_')]
    depth_keys.sort()

    if depth_keys:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        ps = [experiments[k]['config']['p'] for k in depth_keys]
        sim_valid = [experiments[k]['simulator']['valid_rate'] * 100 for k in depth_keys]

        backend_name = list(experiments[depth_keys[0]].get('hardware', {}).keys())
        if backend_name:
            bn = backend_name[0]
            hw_valid = [experiments[k]['hardware'].get(bn, {}).get('valid_rate', 0) * 100
                        for k in depth_keys]
            hw_depths = [experiments[k]['hardware'].get(bn, {}).get('transpiled_depth', 0)
                         for k in depth_keys]
        else:
            hw_valid = [0] * len(ps)
            hw_depths = [0] * len(ps)

        x = np.arange(len(ps))
        width = 0.35
        ax1.bar(x - width/2, sim_valid, width, label='Simulator', color='#9C27B0', alpha=0.8)
        ax1.bar(x + width/2, hw_valid, width, label=f'Hardware ({bn})', color='#E91E63', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'p={p}' for p in ps])
        ax1.set_ylabel('Valid Assignments (%)', fontsize=11)
        ax1.set_title('(a) Valid Rate vs QAOA Depth')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')

        ax2.bar(x, hw_depths, color='#FF9800', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'p={p}' for p in ps])
        ax2.set_ylabel('Transpiled Depth', fontsize=11)
        ax2.set_title('(b) Hardware Circuit Depth')
        ax2.grid(alpha=0.3, axis='y')

        plt.suptitle('QAOA Depth Analysis on Real Quantum Hardware',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig11_hw_depth.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/fig11_hw_depth.png")
        plt.close()

    # === Figure 12: Cross-Backend Comparison ===
    # Use the 12-qubit scaling experiment
    if 'scaling_12q' in experiments:
        hw_data = experiments['scaling_12q'].get('hardware', {})
        if len(hw_data) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

            bn_list = list(hw_data.keys())
            bn_valid = [hw_data[bn].get('valid_rate', 0) * 100 for bn in bn_list]
            bn_costs = [hw_data[bn].get('best_valid_cost') or 0 for bn in bn_list]
            bn_depths = [hw_data[bn].get('transpiled_depth', 0) for bn in bn_list]

            colors = ['#E91E63', '#00BCD4', '#FF9800']

            ax1.bar(range(len(bn_list)), bn_valid, color=colors[:len(bn_list)])
            ax1.set_xticks(range(len(bn_list)))
            ax1.set_xticklabels([bn.replace('ibm_', '') for bn in bn_list], fontsize=10)
            ax1.set_ylabel('Valid Assignments (%)', fontsize=11)
            ax1.set_title('(a) Constraint Satisfaction by Backend')
            for i, v in enumerate(bn_valid):
                ax1.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)
            ax1.grid(alpha=0.3, axis='y')

            ax2.bar(range(len(bn_list)), bn_depths, color=colors[:len(bn_list)])
            ax2.set_xticks(range(len(bn_list)))
            ax2.set_xticklabels([bn.replace('ibm_', '') for bn in bn_list], fontsize=10)
            ax2.set_ylabel('Transpiled Circuit Depth', fontsize=11)
            ax2.set_title('(b) Transpilation Depth by Backend')
            ax2.grid(alpha=0.3, axis='y')

            plt.suptitle('Cross-Backend Reproducibility (12-qubit QAOA, 4000 shots)',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/fig12_cross_backend.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir}/fig12_cross_backend.png")
            plt.close()

    # === Figure 13: Summary Comparison (all methods, all hardware) ===
    fig, ax = plt.subplots(figsize=(10, 6))

    summary_data = []
    if 'scaling_12q' in experiments:
        exp = experiments['scaling_12q']
        # Classical
        summary_data.append(('Greedy\n(Production)', exp['classical']['greedy']['cost'],
                            exp['classical']['greedy']['quality'] * 100, '#2196F3'))
        summary_data.append(('Simulated\nAnnealing', exp['classical']['sa']['cost'],
                            exp['classical']['sa']['quality'] * 100, '#FF9800'))
        # Simulator
        sim_c = exp['simulator'].get('best_valid_cost') or 0
        sim_q = exp['simulator'].get('best_valid_quality', 0) * 100
        summary_data.append(('QAOA\n(Simulator)', sim_c, sim_q, '#9C27B0'))
        # Hardware backends
        for bn, hw in exp.get('hardware', {}).items():
            if isinstance(hw, dict) and 'error' not in hw:
                hw_c = hw.get('best_valid_cost') or 0
                hw_q = hw.get('best_valid_quality', 0) * 100
                short_name = bn.replace('ibm_', '')
                summary_data.append((f'QAOA\n({short_name})', hw_c, hw_q, '#E91E63'))

    if summary_data:
        labels = [d[0] for d in summary_data]
        costs = [d[1] for d in summary_data]
        qualities = [d[2] for d in summary_data]
        colors = [d[3] for d in summary_data]

        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x - width/2, costs, width, label='Best Cost (USD)', color=colors, alpha=0.8)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, qualities, width, label='Quality (%)',
                        color=colors, alpha=0.4, edgecolor=colors, linewidth=2, hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Best Routing Cost (USD)', fontsize=11)
        ax2.set_ylabel('Quality Satisfaction (%)', fontsize=11)
        ax2.set_ylim(0, 110)

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    ax.set_title('Complete Method Comparison: Classical vs Quantum (12-qubit LCRP)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig13_full_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig13_full_comparison.png")
    plt.close()


def print_summary_table(results):
    """Print publication-ready summary table."""
    print("\n" + "=" * 90)
    print("PUBLICATION SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Experiment':>18} | {'Source':>15} | {'Valid %':>8} | {'Best Cost':>10} | "
          f"{'Quality':>8} | {'Depth':>6}")
    print("-" * 90)

    for exp_key, exp in results['experiments'].items():
        label = exp_key.replace('_', ' ').title()

        # Simulator
        sim = exp.get('simulator', {})
        bc = f"${sim.get('best_valid_cost', 0):.2f}" if sim.get('best_valid_cost') else "N/A"
        print(f"{label:>18} | {'Simulator':>15} | {sim.get('valid_rate', 0):>7.1%} | "
              f"{bc:>10} | {sim.get('best_valid_quality', 0):>7.0%} | {'---':>6}")

        # Hardware
        for bn, hw in exp.get('hardware', {}).items():
            if isinstance(hw, dict) and 'error' not in hw:
                bc = f"${hw.get('best_valid_cost', 0):.2f}" if hw.get('best_valid_cost') else "N/A"
                short = bn.replace('ibm_', '')
                print(f"{'':>18} | {short:>15} | {hw.get('valid_rate', 0):>7.1%} | "
                      f"{bc:>10} | {hw.get('best_valid_quality', 0):>7.0%} | "
                      f"{hw.get('transpiled_depth', 0):>6}")

    total_qpu = results['metadata'].get('total_qpu_seconds', 0)
    print(f"\nTotal QPU time used: {total_qpu:.0f}s ({total_qpu/60:.1f} min)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    output_dir = "../results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IBM Quantum Hardware — Comprehensive Experiment Suite")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load optimized parameters from simulator pre-optimization
    params_file = Path("../results/optimized_params.json")
    if params_file.exists():
        with open(params_file) as f:
            optimized_params = json.load(f)
        print(f"Loaded optimized params for {len(optimized_params)} configs")
    else:
        print("WARNING: No optimized_params.json found. Using defaults.")
        print("Run optimize_params.py first for better results.")
        optimized_params = {}

    # Connect to IBM
    service = QiskitRuntimeService()
    backends_available = [b.name for b in service.backends(operational=True)]
    print(f"Available backends: {backends_available}")

    # Use all 3 backends for cross-comparison
    backends = backends_available[:3]

    # Run all experiments
    results = run_all_experiments(service, backends, optimized_params, n_shots=4000)

    # Save raw results (without numpy arrays)
    results_clean = json.loads(json.dumps(results, default=str))
    for exp_key in results_clean['experiments']:
        exp = results_clean['experiments'][exp_key]
        if 'simulator' in exp:
            exp['simulator'].pop('cost_distribution', None)
            exp['simulator'].pop('quality_distribution', None)
        for bn in exp.get('hardware', {}):
            if isinstance(exp['hardware'][bn], dict):
                exp['hardware'][bn].pop('cost_distribution', None)
                exp['hardware'][bn].pop('quality_distribution', None)

    with open(f"{output_dir}/hardware_full_results.json", 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nSaved: {output_dir}/hardware_full_results.json")

    # Generate figures
    generate_all_figures(results, output_dir)

    # Print summary
    print_summary_table(results)

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 70)
