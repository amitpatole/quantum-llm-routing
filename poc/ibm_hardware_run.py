#!/usr/bin/env python3
"""
Run LLM Routing QAOA on Real IBM Quantum Hardware
====================================================

Converts the Cirq QAOA circuit to Qiskit, submits to IBM Eagle/Heron
processor, and compares noisy real-hardware results with simulator.

Prerequisites:
    1. IBM Quantum account (https://quantum.ibm.com)
    2. IBM Cloud API key (from IBM Cloud > Manage > Access > API keys)
    3. Instance CRN (from IBM Quantum dashboard)

Usage:
    # First time: save credentials
    python ibm_hardware_run.py --setup --token YOUR_API_KEY

    # Run experiment
    python ibm_hardware_run.py

Author: Amit Patole
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Cirq (for building circuit)
import cirq

# Qiskit (for IBM hardware submission)
from qiskit import qasm2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Our QAOA implementation
from llm_routing_qaoa import (
    MODELS, Task, RoutingResult,
    generate_task_batch, build_cost_matrix,
    build_quality_penalty, build_latency_penalty,
    solve_greedy, solve_simulated_annealing, solve_brute_force,
)


# IBM Quantum instance CRN
IBM_CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/ccdd4da94a194ccea3b895d6c057e15d:294c80b0-941a-49ba-8bc0-2dbc6a867f12::"


def setup_ibm_account(token: str):
    """Save IBM Quantum credentials for future use."""
    print("Saving IBM Quantum credentials...")
    QiskitRuntimeService.save_account(
        channel="ibm_cloud",
        token=token,
        instance=IBM_CRN,
        overwrite=True,
        set_as_default=True,
    )
    print("Credentials saved successfully.")

    # Verify connection
    service = QiskitRuntimeService()
    backends = service.backends()
    print(f"\nAvailable backends ({len(backends)}):")
    for b in backends:
        print(f"  - {b.name}: {b.num_qubits} qubits, status={b.status().status_msg}")


def list_backends():
    """List all available IBM Quantum backends and their status."""
    service = QiskitRuntimeService()
    backends = service.backends()
    print(f"\nAvailable IBM Quantum Backends ({len(backends)}):")
    print(f"{'Name':>25} | {'Qubits':>6} | {'Status':>12} | {'Queue':>6}")
    print("-" * 65)
    for b in backends:
        status = b.status()
        print(f"{b.name:>25} | {b.num_qubits:>6} | {status.status_msg:>12} | "
              f"{status.pending_jobs:>6}")


def build_qaoa_circuit_for_ibm(tasks, models, budget, p_layers=2,
                                gammas=None, betas=None):
    """Build a QAOA circuit in Cirq, optimized for IBM hardware export.

    Uses pre-optimized parameters from simulator runs to avoid
    running the variational loop on hardware (too expensive).
    """
    n, m = len(tasks), len(models)
    num_qubits = n * m

    # Default parameters (from simulator optimization)
    if gammas is None:
        gammas = [0.5] * p_layers
    if betas is None:
        betas = [0.3] * p_layers

    # Build cost coefficients
    C = build_cost_matrix(tasks, models)
    P = build_quality_penalty(tasks, models)
    L = build_latency_penalty(tasks, models)
    max_cost = np.max(C) if np.max(C) > 0 else 1.0

    # Linear terms
    h = np.zeros(num_qubits)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            h[idx] = C[i, j] / max_cost + 40.0 * P[i, j] + 3.0 * L[i, j]

    # Coupling terms (one-hot constraint)
    couplings = []
    for i in range(n):
        for j in range(m):
            for k in range(j + 1, m):
                couplings.append((i * m + j, i * m + k, 50.0))

    # Build Cirq circuit
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit([cirq.H(q) for q in qubits])

    for l in range(p_layers):
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
    """Convert a Cirq circuit to Qiskit via OpenQASM 2.0.

    Handles sx/sxdg gates that Cirq emits for ZZ decomposition
    but are not in qelib1.inc — we inject gate definitions into the QASM.
    """
    qasm_str = cirq.qasm(cirq_circuit, args=cirq.QasmArgs(version="2.0"))

    # Inject sx/sxdg gate definitions after the include line
    # sx = sqrt(X), sxdg = sqrt(X)-dagger
    gate_defs = (
        '\n// Custom gate definitions for Cirq compatibility\n'
        'gate sx a { rz(-pi/2) a; ry(pi/2) a; rz(pi/2) a; }\n'
        'gate sxdg a { rz(-pi/2) a; ry(-pi/2) a; rz(pi/2) a; }\n'
    )
    qasm_str = qasm_str.replace(
        'include "qelib1.inc";',
        'include "qelib1.inc";' + gate_defs,
    )

    qiskit_circuit = qasm2.loads(qasm_str)
    return qiskit_circuit, qasm_str


def run_on_simulator_cirq(cirq_circuit, n_shots=4000):
    """Run on Cirq simulator for comparison baseline."""
    print(f"\n[Simulator] Running {n_shots} shots on Cirq simulator...")
    t0 = time.time()
    simulator = cirq.Simulator()
    result = simulator.run(cirq_circuit, repetitions=n_shots)
    elapsed = time.time() - t0
    measurements = result.measurements['result']
    print(f"[Simulator] Done in {elapsed:.1f}s")
    return measurements


def run_on_ibm_hardware(qiskit_circuit, backend_name=None, n_shots=4000):
    """Submit circuit to IBM Quantum hardware and wait for results."""
    service = QiskitRuntimeService()

    # Pick the least-busy backend if not specified
    if backend_name is None:
        backends = service.backends(
            min_num_qubits=qiskit_circuit.num_qubits,
            operational=True,
        )
        if not backends:
            print("ERROR: No suitable backends found!")
            return None

        # Sort by queue length
        backends.sort(key=lambda b: b.status().pending_jobs)
        backend = backends[0]
        print(f"\n[Hardware] Auto-selected: {backend.name} "
              f"({backend.num_qubits} qubits, {backend.status().pending_jobs} jobs in queue)")
    else:
        backend = service.backend(backend_name)
        print(f"\n[Hardware] Using: {backend.name} ({backend.num_qubits} qubits)")

    # Transpile for the target backend
    print(f"[Hardware] Transpiling circuit for {backend.name}...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qiskit_circuit)

    print(f"[Hardware] Transpiled: {isa_circuit.num_qubits} qubits, "
          f"{isa_circuit.depth()} depth, {isa_circuit.count_ops()} gates")

    # Submit
    print(f"[Hardware] Submitting {n_shots} shots...")
    sampler = SamplerV2(backend)
    job = sampler.run([isa_circuit], shots=n_shots)
    print(f"[Hardware] Job ID: {job.job_id()}")
    print(f"[Hardware] Waiting for results (this may take minutes in queue)...")

    t0 = time.time()
    result = job.result()
    elapsed = time.time() - t0
    print(f"[Hardware] Results received in {elapsed:.1f}s")

    return result, backend.name, isa_circuit


def decode_assignment(bitstring, tasks, models):
    """Decode a binary measurement into task-to-model assignments."""
    n, m = len(tasks), len(models)
    X = np.array(bitstring).reshape(n, m)

    assignments = {}
    total_cost = 0.0
    quality_met = 0

    for i, task in enumerate(tasks):
        assigned = np.where(X[i] == 1)[0]
        if len(assigned) > 0:
            j = assigned[0]
            model = models[j]
            assignments[task.task_id] = model.name
            total_cost += model.cost_per_1k_tokens * task.estimated_tokens / 1000
            if model.quality_score >= task.min_quality:
                quality_met += 1
        else:
            assignments[task.task_id] = "UNASSIGNED"

    return assignments, total_cost, quality_met / max(1, n)


def analyze_results(measurements, tasks, models, source_label):
    """Analyze measurement results and find best assignment."""
    print(f"\n{'='*60}")
    print(f"Analysis: {source_label}")
    print(f"{'='*60}")

    n_shots = len(measurements)
    best_cost = float('inf')
    best_assignment = None
    best_quality = 0.0

    cost_distribution = []
    quality_distribution = []
    valid_count = 0

    for m_bits in measurements:
        assignments, cost, quality = decode_assignment(m_bits, tasks, models)
        cost_distribution.append(cost)
        quality_distribution.append(quality)

        # Check if valid (each task assigned exactly one model)
        n, m = len(tasks), len(models)
        X = np.array(m_bits).reshape(n, m)
        is_valid = all(np.sum(X[i]) == 1 for i in range(n))
        if is_valid:
            valid_count += 1

        if cost < best_cost and is_valid:
            best_cost = cost
            best_assignment = assignments
            best_quality = quality

    print(f"  Total shots: {n_shots}")
    print(f"  Valid assignments: {valid_count}/{n_shots} ({valid_count/n_shots:.1%})")
    print(f"  Cost — mean: ${np.mean(cost_distribution):.4f}, "
          f"std: ${np.std(cost_distribution):.4f}, "
          f"min: ${np.min(cost_distribution):.4f}, max: ${np.max(cost_distribution):.4f}")
    print(f"  Quality — mean: {np.mean(quality_distribution):.2%}")

    if best_assignment:
        print(f"\n  Best valid assignment (cost: ${best_cost:.4f}, quality: {best_quality:.0%}):")
        for tid, mname in best_assignment.items():
            print(f"    {tid} -> {mname}")

    return {
        'source': source_label,
        'n_shots': n_shots,
        'valid_fraction': valid_count / n_shots,
        'best_cost': best_cost,
        'best_quality': best_quality,
        'best_assignment': best_assignment,
        'mean_cost': float(np.mean(cost_distribution)),
        'std_cost': float(np.std(cost_distribution)),
        'cost_distribution': [float(c) for c in cost_distribution],
        'quality_distribution': [float(q) for q in quality_distribution],
    }


def plot_comparison(sim_analysis, hw_analysis, classical_results, output_dir):
    """Generate comparison plots: simulator vs hardware vs classical."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Cost distribution histogram
    ax = axes[0, 0]
    bins = np.linspace(0, max(max(sim_analysis['cost_distribution']),
                              max(hw_analysis['cost_distribution'])) * 1.1, 30)
    ax.hist(sim_analysis['cost_distribution'], bins=bins, alpha=0.6,
            label='Simulator', color='#2196F3', density=True)
    ax.hist(hw_analysis['cost_distribution'], bins=bins, alpha=0.6,
            label='Real Hardware', color='#E91E63', density=True)
    for name, r in classical_results.items():
        ax.axvline(r.total_cost, color='green' if name == 'greedy' else 'orange',
                   linestyle='--', linewidth=2, label=f'{name}: ${r.total_cost:.2f}')
    ax.set_xlabel('Routing Cost (USD)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Cost Distribution: Simulator vs Hardware')
    ax.legend(fontsize=8)

    # 2. Quality distribution
    ax = axes[0, 1]
    sim_q = [q * 100 for q in sim_analysis['quality_distribution']]
    hw_q = [q * 100 for q in hw_analysis['quality_distribution']]
    ax.hist(sim_q, bins=20, alpha=0.6, label='Simulator', color='#2196F3', density=True)
    ax.hist(hw_q, bins=20, alpha=0.6, label='Real Hardware', color='#E91E63', density=True)
    ax.set_xlabel('Quality Satisfaction (%)')
    ax.set_ylabel('Density')
    ax.set_title('(b) Quality Distribution: Simulator vs Hardware')
    ax.legend(fontsize=8)

    # 3. Summary bar chart
    ax = axes[1, 0]
    methods = ['Greedy', 'Sim. Anneal.', 'QAOA\n(Simulator)', 'QAOA\n(Hardware)']
    costs = [
        classical_results['greedy'].total_cost,
        classical_results['sa'].total_cost,
        sim_analysis['best_cost'],
        hw_analysis['best_cost'],
    ]
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#E91E63']
    ax.bar(methods, costs, color=colors)
    ax.set_ylabel('Best Cost Found (USD)')
    ax.set_title('(c) Best Solution: All Methods')
    for i, (m, c) in enumerate(zip(methods, costs)):
        ax.text(i, c + 0.3, f'${c:.2f}', ha='center', fontsize=9)

    # 4. Valid assignment rates
    ax = axes[1, 1]
    labels = ['Simulator', 'Real Hardware']
    valid_rates = [sim_analysis['valid_fraction'] * 100,
                   hw_analysis['valid_fraction'] * 100]
    bars = ax.bar(labels, valid_rates, color=['#9C27B0', '#E91E63'])
    ax.set_ylabel('Valid Assignments (%)')
    ax.set_title('(d) Constraint Satisfaction Rate')
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, valid_rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                ha='center', fontsize=11)

    plt.suptitle('QAOA LLM Routing: Simulator vs Real Quantum Hardware (IBM)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f"{output_dir}/fig9_hardware_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run QAOA on IBM Quantum Hardware')
    parser.add_argument('--setup', action='store_true', help='Save IBM credentials')
    parser.add_argument('--token', type=str, help='IBM Cloud API key')
    parser.add_argument('--backend', type=str, default=None, help='IBM backend name')
    parser.add_argument('--list-backends', action='store_true', help='List available backends')
    parser.add_argument('--shots', type=int, default=4000, help='Number of shots')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build circuit and export QASM only, do not submit to hardware')
    args = parser.parse_args()

    # --- Setup ---
    if args.setup:
        if not args.token:
            print("ERROR: --token required with --setup")
            print("Get your API key from: IBM Cloud > Manage > Access (IAM) > API keys")
            sys.exit(1)
        setup_ibm_account(args.token)
        return

    if args.list_backends:
        list_backends()
        return

    # --- Build experiment ---
    print("=" * 70)
    print("QAOA LLM Routing — IBM Quantum Hardware Experiment")
    print("=" * 70)

    # Problem: 4 tasks x 3 models = 12 qubits (fits comfortably on 127-qubit Eagle)
    tasks = generate_task_batch(4, seed=42)
    models = MODELS[:3]  # gemma2, haiku, sonnet
    budget = 5.0
    n_shots = args.shots

    print(f"\nProblem: {len(tasks)} tasks x {len(models)} models = {len(tasks)*len(models)} qubits")
    print(f"Tasks: {[(t.task_id, t.complexity) for t in tasks]}")
    print(f"Models: {[m.name for m in models]}")
    print(f"Budget: ${budget:.2f}")
    print(f"Shots: {n_shots}")

    # --- Build Cirq circuit ---
    print("\n--- Building QAOA Circuit (Cirq) ---")
    cirq_circuit = build_qaoa_circuit_for_ibm(tasks, models, budget, p_layers=2)
    print(f"Cirq circuit: {len(cirq_circuit)} moments, "
          f"{sum(1 for _ in cirq_circuit.all_operations())} operations")

    # --- Convert to Qiskit ---
    print("\n--- Converting Cirq -> Qiskit (via OpenQASM) ---")
    qiskit_circuit, qasm_str = cirq_to_qiskit(cirq_circuit)
    print(f"Qiskit circuit: {qiskit_circuit.num_qubits} qubits, "
          f"{qiskit_circuit.depth()} depth")

    # Save QASM for inspection
    qasm_path = Path("../results/qaoa_circuit.qasm")
    qasm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qasm_path, 'w') as f:
        f.write(qasm_str)
    print(f"Saved QASM: {qasm_path}")

    if args.dry_run:
        print("\n--- DRY RUN: Circuit built and exported, not submitting to hardware ---")
        print(f"\nQASM preview (first 20 lines):")
        for line in qasm_str.split('\n')[:20]:
            print(f"  {line}")
        return

    # --- Run classical baselines ---
    print("\n--- Classical Baselines ---")
    greedy_result = solve_greedy(tasks, models, budget)
    sa_result = solve_simulated_annealing(tasks, models, budget)
    bf_result = solve_brute_force(tasks, models, budget)
    classical = {'greedy': greedy_result, 'sa': sa_result}
    if bf_result:
        classical['brute_force'] = bf_result

    print(f"  Greedy:      ${greedy_result.total_cost:.4f} ({greedy_result.quality_satisfied:.0%})")
    print(f"  Sim Anneal:  ${sa_result.total_cost:.4f} ({sa_result.quality_satisfied:.0%})")
    if bf_result:
        print(f"  Brute Force: ${bf_result.total_cost:.4f} ({bf_result.quality_satisfied:.0%})")

    # --- Run on Cirq simulator ---
    sim_measurements = run_on_simulator_cirq(cirq_circuit, n_shots=n_shots)
    sim_analysis = analyze_results(sim_measurements, tasks, models, "Cirq Simulator")

    # --- Run on IBM Hardware ---
    print("\n--- Submitting to IBM Quantum Hardware ---")
    hw_output = run_on_ibm_hardware(
        qiskit_circuit, backend_name=args.backend, n_shots=n_shots)

    if hw_output is None:
        print("ERROR: Hardware submission failed!")
        return

    hw_result, backend_name, transpiled = hw_output

    # Extract measurements from Qiskit SamplerV2 result
    pub_result = hw_result[0]
    # SamplerV2 stores results under data.<creg_name>
    # Our QASM register is "m_result", try common names
    bitarray = None
    for attr in ['m_result', 'meas', 'c']:
        if hasattr(pub_result.data, attr):
            bitarray = getattr(pub_result.data, attr)
            break
    if bitarray is None:
        # Fallback: iterate data attributes
        for attr in dir(pub_result.data):
            if not attr.startswith('_'):
                bitarray = getattr(pub_result.data, attr)
                break

    # Convert BitArray to numpy array
    bitstrings = bitarray.get_bitstrings()
    hw_measurements = np.array([[int(b) for b in bs] for bs in bitstrings])

    hw_analysis = analyze_results(hw_measurements, tasks, models,
                                  f"IBM Hardware ({backend_name})")

    # --- Generate comparison plots ---
    plot_comparison(sim_analysis, hw_analysis, classical, "../results")

    # --- Save all results ---
    all_results = {
        'experiment': {
            'n_tasks': len(tasks),
            'n_models': len(models),
            'n_qubits': len(tasks) * len(models),
            'budget': budget,
            'n_shots': n_shots,
            'backend': backend_name,
            'transpiled_depth': transpiled.depth(),
        },
        'classical': {
            'greedy': {'cost': greedy_result.total_cost,
                       'quality': greedy_result.quality_satisfied},
            'sa': {'cost': sa_result.total_cost,
                   'quality': sa_result.quality_satisfied},
        },
        'simulator': {
            'best_cost': sim_analysis['best_cost'],
            'best_quality': sim_analysis['best_quality'],
            'valid_fraction': sim_analysis['valid_fraction'],
            'mean_cost': sim_analysis['mean_cost'],
        },
        'hardware': {
            'best_cost': hw_analysis['best_cost'],
            'best_quality': hw_analysis['best_quality'],
            'valid_fraction': hw_analysis['valid_fraction'],
            'mean_cost': hw_analysis['mean_cost'],
            'backend': backend_name,
        },
    }

    results_path = Path("../results/ibm_hardware_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE — Summary")
    print("=" * 70)
    print(f"\n{'Method':>25} | {'Best Cost':>10} | {'Quality':>8} | {'Valid %':>8}")
    print("-" * 60)
    print(f"{'Greedy (Production)':>25} | ${greedy_result.total_cost:>8.4f} | "
          f"{greedy_result.quality_satisfied:>7.0%} | {'100.0%':>8}")
    print(f"{'Simulated Annealing':>25} | ${sa_result.total_cost:>8.4f} | "
          f"{sa_result.quality_satisfied:>7.0%} | {'100.0%':>8}")
    print(f"{'QAOA (Simulator)':>25} | ${sim_analysis['best_cost']:>8.4f} | "
          f"{sim_analysis['best_quality']:>7.0%} | {sim_analysis['valid_fraction']:>7.1%}")
    print(f"{'QAOA (IBM ' + backend_name + ')':>25} | ${hw_analysis['best_cost']:>8.4f} | "
          f"{hw_analysis['best_quality']:>7.0%} | {hw_analysis['valid_fraction']:>7.1%}")
    print("=" * 70)


if __name__ == '__main__':
    main()
