"""Circuit-level dynamic-Rodeo NESS filter on a noiseless ``AerSimulator``.

This module compiles the phase-symmetric Rodeo zero-sector filter of the paper as
an explicit Qiskit circuit with **mid-circuit measurement and conditional skipping**
(measurement-conditioned cycles that abort the remaining schedule on a failure), and
runs it on the noiseless :class:`qiskit_aer.AerSimulator`.  It reproduces the
circuit-level validation figures (Appendix E): the convergence of the post-selected
``<sigma_z>`` estimate to the exact NESS value (Fig. 8) and the gate-level early-abort
cycle saving (Fig. 9).

The circuit builders and the post-selection / ratio reconstruction are taken verbatim
from the hardware-facing sweep notebook (``qiskit_runtime_dynamic_rodeo_qpe_sweep_fez``);
the only change is that the embedding ``M`` is obtained from
:func:`rodeo_ness.liouvillian.single_spin_liouvillian` (verified bit-identical to the
notebook's dense model), so this module depends only on the local package.

Requires the optional ``circuit`` extra::

    pip install "rodeo_ness[circuit]"
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

from .liouvillian import single_spin_liouvillian, hermitian_embedding, steady_state

I2 = np.eye(2, dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def embedding_M(h: float) -> np.ndarray:
    """Hermitian embedding ``M`` of the single-spin Liouvillian (package model)."""
    return hermitian_embedding(single_spin_liouvillian(h))


def exact_sz(h: float = 0.5) -> float:
    """Exact NESS value ``<sigma_z>_ss`` (``-1/3`` at ``h=0.5``)."""
    rho = steady_state(single_spin_liouvillian(h))
    return float(np.real(np.trace(_SZ @ rho)))


def vec(rho: np.ndarray) -> np.ndarray:
    """Column-stacking vectorization: index = row + d*col."""
    return np.asarray(rho, dtype=complex).reshape(-1, order='F')


def normalized_input_xi() -> np.ndarray:
    """|xi> = (|0>|I_norm> + |1>|0><0|)/sqrt(2), as an 8-vector.

    Branch is the most significant qubit in the matrix block ordering.
    Qiskit little-endian register order is [row, col, branch].
    """
    vI = vec(I2) / np.sqrt(2.0)  # normalized Bell-state identity vector
    rho0 = np.zeros((2, 2), dtype=complex)
    rho0[0, 0] = 1.0
    v0 = vec(rho0)
    xi = np.concatenate([vI, v0]) / np.sqrt(2.0)
    assert np.allclose(np.linalg.norm(xi), 1.0)
    return xi


def deterministic_vdc_schedule(n: int, g: float = 0.5, A: float = 3.0, base: int = 2) -> np.ndarray:
    """Low-discrepancy van-der-Corput times in [0, A/g].

    This is an empirical hardware-friendly schedule used for the toy benchmark.
    """
    def vdc(k, base=2):
        x = 0.0
        denom = 1.0
        while k:
            k, rem = divmod(k, base)
            denom *= base
            x += rem / denom
        return x
    return np.array([(A / g) * vdc(k + 1, base=base) for k in range(n)], dtype=float)


def expm_from_hermitian(H: np.ndarray, time: float) -> np.ndarray:
    """Return exp(-i H time) using an eigendecomposition of Hermitian H."""
    evals, evecs = np.linalg.eigh(H)
    U = evecs @ np.diag(np.exp(-1j * time * evals)) @ evecs.conj().T
    return np.asarray(U, dtype=complex)


def M_evolution_gate(h: float, time: float, label: str | None = None) -> UnitaryGate:
    """Dense exact gate for exp(-i M time) on the three M-register qubits."""
    U = expm_from_hermitian(embedding_M(h), time)
    return UnitaryGate(U, label=label or f'exp(-iMt={time:.3g})')


def append_symmetric_rodeo_cycle_body(qc: QuantumCircuit, ancilla, m_qubits, h: float, t: float):
    """Append one symmetric Rodeo cycle body without measurement.

    The ancilla is assumed to start in |0>.  Measuring it in the computational basis after
    the final H gives success outcome 0 and implements cos(M t/2) on the M register.
    """
    U_plus = M_evolution_gate(h, -t / 2.0, label=f'exp(+iM{t/2:.2g})')
    U_minus = M_evolution_gate(h, +t / 2.0, label=f'exp(-iM{t/2:.2g})')

    qc.h(ancilla)
    # Apply U_plus to the original |0> arm by flipping the control.
    qc.x(ancilla)
    qc.append(U_plus.control(1), [ancilla] + list(m_qubits))
    qc.x(ancilla)
    # Apply U_minus to the original |1> arm.
    qc.append(U_minus.control(1), [ancilla] + list(m_qubits))
    qc.h(ancilla)


def append_mid_measure(qc: QuantumCircuit, qubit, clbit):
    """Mid-circuit measurement helper."""
    qc.measure(qubit, clbit)


def build_rodeo_dynamic_base_circuit(h: float, times: np.ndarray) -> tuple[QuantumCircuit, list, int]:
    """Dynamic Rodeo base circuit with one reused ancilla.

    Classical bits in one register `meas`:
      meas[0:n]       = Rodeo cycle success bits, all must be 0.
      meas[n:n+3]     = final data-register bits for the Pauli readout.

    The conditional block contains only gates, not measurements or reset, matching the
    current IBM Runtime dynamic-circuit restrictions.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    if n + 3 > 32:
        raise ValueError('This simple demo conditions on the full classical register, so n+3 must be <= 32.')

    m = QuantumRegister(3, 'm')      # [row, col, branch]
    r = QuantumRegister(1, 'r')      # reused Rodeo ancilla
    c = ClassicalRegister(n + 3, 'meas')
    qc = QuantumCircuit(m, r, c, name=f'rodeo_dynamic_n{n}')

    qc.append(StatePreparation(normalized_input_xi(), label='prep_xi'), list(m))
    for k, t in enumerate(times):
        if k == 0:
            append_symmetric_rodeo_cycle_body(qc, r[0], list(m), h, float(t))
        else:
            # Execute the next expensive controlled evolutions only if all previous success bits are zero.
            # Future bits and final data bits are still initialized to zero at this point.
            with qc.if_test((c, 0)):
                append_symmetric_rodeo_cycle_body(qc, r[0], list(m), h, float(t))
        # Measurement must stay outside the if block.  If a previous failure occurred, this records
        # the still-collapsed failure ancilla again, and the shot is excluded by postselection.
        append_mid_measure(qc, r[0], c[k])
    return qc, list(m), n


def append_terminal_xiz_readout(qc: QuantumCircuit, m_qubits: list, n_success: int):
    """Append terminal XIZ-basis readout.

    With qubit order [row, col, branch], this basis gives both
      R_I = <X_branch>
      R_z = <X_branch Z_row>
    from the same counts.  Their ratio estimates <sigma_z>.
    """
    c = qc.cregs[0]
    qc.h(m_qubits[2])  # branch qubit in X basis
    for q_index, q in enumerate(m_qubits):
        qc.measure(q, c[n_success + q_index])


def parse_count_bitstring(bitstring: str) -> str:
    """Return little-endian classical bit string: output[0] is classical bit 0."""
    return bitstring.replace(' ', '')[::-1]


def cycles_executed_from_success_bits(success_bits_little: str) -> int:
    """Number of Rodeo cycle bodies executed in the dynamic circuit for one shot.

    If the first failure is at cycle k, cycles 0..k were executed and the rest skipped.
    If all bits are zero, all cycles were executed.
    """
    n = len(success_bits_little)
    for k, b in enumerate(success_bits_little):
        if b == '1':
            return k + 1
    return n


def reconstruct_I_and_z_from_xiz_counts(counts: dict, n_success: int, denom_tol: float = 0.05) -> dict:
    """Postselect success=0...0 and estimate RI, Rz, and Rz/RI from one XIZ-basis circuit."""
    kept = 0
    rI_sum = 0
    rz_sum = 0
    total = sum(counts.values())
    executed_cycle_sum = 0

    for bitstring, count in counts.items():
        bits = parse_count_bitstring(bitstring)
        success_bits = bits[:n_success]
        if n_success > 0:
            executed_cycle_sum += cycles_executed_from_success_bits(success_bits) * count
        if success_bits != '0' * n_success:
            continue
        kept += count
        row_z_bit = int(bits[n_success + 0])       # q0=row, Z basis
        branch_x_bit = int(bits[n_success + 2])    # q2=branch, X basis after H
        rI_sum += ((-1) ** branch_x_bit) * count
        rz_sum += ((-1) ** (branch_x_bit ^ row_z_bit)) * count

    if kept == 0:
        return {
            'kept': 0, 'shots': total, 'p_success': 0.0,
            'R_I': np.nan, 'R_z': np.nan, 'sz_estimate': np.nan,
            'avg_executed_cycles': executed_cycle_sum / total if total else np.nan,
            'cycle_saving_vs_static': np.nan,
        }

    R_I = rI_sum / kept
    R_z = rz_sum / kept
    sz_est = np.nan if abs(R_I) < denom_tol else R_z / R_I
    avg_cycles = executed_cycle_sum / total if total and n_success > 0 else np.nan
    saving = 1 - avg_cycles / n_success if n_success > 0 and np.isfinite(avg_cycles) else np.nan
    return {
        'kept': kept,
        'shots': total,
        'p_success': kept / total if total else np.nan,
        'R_I': R_I,
        'R_z': R_z,
        'sz_estimate': sz_est,
        'avg_executed_cycles': avg_cycles,
        'cycle_saving_vs_static': saving,
    }


def shot_error_estimates(counts: dict, n_success: int, denom_tol: float = 0.05) -> dict:
    """Shot-noise standard errors for the post-selected estimates (delta method).

    ``sz_se`` is the standard error of the ratio estimator ``R_z / R_I``, propagated
    from the multinomial shot statistics of the kept (success = 0...0) shots and
    including the R_z--R_I covariance (the two readouts share the same shots).
    ``avg_executed_se`` is the standard error of the mean executed-cycle count over
    all shots.  Both are validated against a bootstrap over the shot counts.
    """
    K = sa = sb = sc = 0          # kept count; sums of a=(-1)^x, b=(-1)^(x^z), c=(-1)^z
    total = n_exec_sum = n_exec_sq = 0
    for bitstring, cnt in counts.items():
        bits = parse_count_bitstring(bitstring)
        total += cnt
        if n_success > 0:
            ce = cycles_executed_from_success_bits(bits[:n_success])
            n_exec_sum += ce * cnt
            n_exec_sq += ce * ce * cnt
        if bits[:n_success] != "0" * n_success:
            continue
        x = int(bits[n_success + 2]); z = int(bits[n_success + 0])
        a = -1 if x else 1
        b = -1 if (x ^ z) else 1
        c = -1 if z else 1
        K += cnt; sa += a * cnt; sb += b * cnt; sc += c * cnt

    out = {"sz_se": np.nan, "avg_executed_se": np.nan}
    if K > 1:
        R_I = sa / K
        R_z = sb / K
        if abs(R_I) >= denom_tol:
            var_a = 1.0 - R_I * R_I
            var_b = 1.0 - R_z * R_z
            cov_ab = sc / K - R_I * R_z        # mean(a*b) = mean((-1)^z) = sc/K
            var_sz = (var_b / R_I ** 2
                      + (R_z ** 2 / R_I ** 4) * var_a
                      - 2.0 * (R_z / R_I ** 3) * cov_ab) / K
            out["sz_se"] = float(np.sqrt(max(var_sz, 0.0)))
    if n_success > 0 and total > 1:
        mean_ce = n_exec_sum / total
        var_ce = max(n_exec_sq / total - mean_ce ** 2, 0.0)
        out["avg_executed_se"] = float(np.sqrt(var_ce / total))
    return out


# --------------------------------------------------------------------------
# High-level noiseless-Aer sweep (Figs. 8 and 9)
# --------------------------------------------------------------------------
def _single_run(h, n_values, shots, seed):
    """One noiseless-Aer pass over n_values; returns {n: stats-tuple}."""
    sim = AerSimulator(seed_simulator=seed)
    sampler = AerSamplerV2(options={"backend_options": {"seed_simulator": seed}})
    out = {}
    for n in n_values:
        times = deterministic_vdc_schedule(n)
        base, m_qubits, nsucc = build_rodeo_dynamic_base_circuit(h, times)
        append_terminal_xiz_readout(base, m_qubits, nsucc)
        counts = sampler.run([transpile(base, sim)], shots=shots).result()[0].data.meas.get_counts()
        s = reconstruct_I_and_z_from_xiz_counts(counts, nsucc)
        se = shot_error_estimates(counts, nsucc)
        out[n] = dict(sz=s["sz_estimate"], R_I=s["R_I"], avg=s["avg_executed_cycles"],
                      sav=s["cycle_saving_vs_static"], p=s["p_success"],
                      sz_se=se["sz_se"], avg_se=se["avg_executed_se"])
    return out


def run_convergence_sweep(h: float = 0.5, n_values=range(1, 13), shots: int = 40000,
                          seed: int = 1234, n_repeats: int = 1,
                          progress_path: str | None = None) -> list[dict]:
    """Run the dynamic Rodeo circuit on a noiseless ``AerSimulator`` for each ``n``.

    Returns one record per cycle count with the post-selected ``<sigma_z>`` estimate,
    its absolute error versus the exact NESS, and the dynamic early-abort statistics.

    Error bars
    ----------
    * ``n_repeats == 1`` (default): ``sz_err`` is the analytic (delta-method) shot-noise
      standard error of the single run -- fast, good for a quick check.
    * ``n_repeats > 1``: the whole sweep is repeated with ``n_repeats`` independent,
      well-separated simulator seeds drawn from a generator seeded by ``seed`` (consecutive
      Aer seeds are correlated, so this is required for genuine independent realisations);
      the reported value is the per-``n`` mean and ``sz_err`` is the **empirical
      run-to-run standard deviation** -- the directly measured shot-noise spread. This
      is the more empirical / publication-grade estimate (and validates the delta method).
      Likewise ``avg_executed_err`` and ``cycle_saving_err`` become empirical std's.
    """
    n_values = list(n_values)
    sz_ss = exact_sz(h)

    if n_repeats <= 1:
        run = _single_run(h, n_values, shots, seed)
        out = []
        for n in n_values:
            r = run[n]
            out.append({
                "n": n, "R_I": r["R_I"], "sz_estimate": r["sz"],
                "abs_error_sz": abs(r["sz"] - sz_ss) if np.isfinite(r["sz"]) else np.nan,
                "sz_err": r["sz_se"], "static_cycles": n,
                "avg_executed_cycles": r["avg"], "avg_executed_err": r["avg_se"],
                "cycle_saving": r["sav"],
                "cycle_saving_err": (r["avg_se"] / n) if (n > 0 and np.isfinite(r["avg_se"])) else np.nan,
                "p_success": r["p"], "n_repeats": 1,
            })
        return out

    # Independent realisations require *well-separated* simulator seeds: consecutive
    # AerSimulator seeds (seed, seed+1, ...) produce strongly correlated shot samples,
    # which collapses the run-to-run spread by ~100x. We therefore draw well-mixed
    # seeds from a generator seeded by ``seed`` -- independent realisations, still
    # fully reproducible. ``progress_path`` checkpoints the raw arrays after every
    # repeat so a long run can be plotted/recovered partway through.
    rng = np.random.default_rng(seed)
    sim_seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=n_repeats)]
    acc = {n: {"sz": [], "avg": [], "sav": []} for n in n_values}
    for rep, s in enumerate(sim_seeds):
        run = _single_run(h, n_values, shots, s)
        for n in n_values:
            acc[n]["sz"].append(run[n]["sz"]); acc[n]["avg"].append(run[n]["avg"]); acc[n]["sav"].append(run[n]["sav"])
        if progress_path:
            import json as _json
            with open(progress_path, "w") as _f:
                _json.dump({"h": h, "shots": shots, "n_values": n_values, "exact": sz_ss,
                            "completed_repeats": rep + 1, "n_repeats": n_repeats, "acc": acc}, _f)

    def msd(a):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return (float(np.mean(a)) if a.size else np.nan,
                float(np.std(a, ddof=1)) if a.size > 1 else np.nan)

    out = []
    for n in n_values:
        szm, szs = msd(acc[n]["sz"]); avgm, avgs = msd(acc[n]["avg"]); savm, savs = msd(acc[n]["sav"])
        out.append({
            "n": n, "sz_estimate": szm, "sz_err": szs,
            "abs_error_sz": abs(szm - sz_ss) if np.isfinite(szm) else np.nan,
            "static_cycles": n, "avg_executed_cycles": avgm, "avg_executed_err": avgs,
            "cycle_saving": savm, "cycle_saving_err": savs, "n_repeats": n_repeats,
        })
    return out


def expected_executed_cycles(h: float, n: int) -> float:
    """Exact expected number of Rodeo cycle bodies executed before the first failed
    success bit (single-spin embedding, deterministic schedule), computed on the
    statevector -- no sampling. The mean cycle saving versus the static schedule is
    ``1 - expected_executed_cycles(h, n) / n``. This is what the noiseless circuit
    estimates; it matches the AerSimulator run to ~1e-3 and lets the saving curve be
    extended to large ``n`` cheaply. The saving rises monotonically and approaches a
    finite asymptote (~43% for h=0.5) very slowly -- it does *not* saturate near 35%.
    """
    M = embedding_M(h)
    psi = np.asarray(normalized_input_xi(), complex)
    psi = psi / np.linalg.norm(psi)
    w, V = np.linalg.eigh(M)
    reach, E = 1.0, 0.0
    for t in deterministic_vdc_schedule(n):
        E += reach
        cosM = (V * np.cos(w * t / 2.0)) @ V.conj().T
        psi2 = cosM @ psi
        p = float(np.vdot(psi2, psi2).real)
        reach *= p
        if p > 1e-300:
            psi = psi2 / np.sqrt(p)
    return E
