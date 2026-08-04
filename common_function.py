"""
Stage 2 additions to common_function.py  (same frozen conventions).

Adds: spin-s operators & edges, eig-based unitary log with branch guard,
the multidegree extraction engine, d-general weight decomposition,
protocol builders (Trotter / Strang / palindrome / 12-pulse witness),
channel-basis projection, symmetric random couplings, and the
convention self-test gate `run_convention_selftests()`.
"""

from itertools import product
from math import factorial

import numpy as np

from common_function import (
    I2, X, Y, Z, kron_all, two_body, commutator,
    antihermitian_part, general_edge_hamiltonian,
    protocol_unitary, protocol_log,
)

CONVENTION_VERSION = "frozen-v1 (chronological: first-acting rightmost; L anti-Hermitian; K = iL)"


# ---------------------------------------------------------------------
# Spin-s operators and edges (Section 5)
# ---------------------------------------------------------------------

def spin_ops(local_dim, normalized=True):
    """Sx, Sy, Sz for spin s = (d-1)/2; normalized=True divides by s."""
    s = (local_dim - 1) / 2
    m = np.arange(s, -s - 1, -1)
    Sz = np.diag(m).astype(complex)
    Sp = np.zeros((local_dim, local_dim), dtype=complex)
    for k in range(local_dim - 1):
        Sp[k, k + 1] = np.sqrt(s * (s + 1) - m[k + 1] * (m[k + 1] + 1))
    Sx = (Sp + Sp.conj().T) / 2
    Sy = (Sp - Sp.conj().T) / (2j)
    if normalized and s > 0:
        Sx, Sy, Sz = Sx / s, Sy / s, Sz / s
    return Sx, Sy, Sz


def embed_two_body(op_i, op_j, site_i, site_j, n_sites, local_dim):
    """d-general two-site embedding (generalizes common_function.two_body)."""
    if site_i == site_j:
        raise ValueError("site_i and site_j must be distinct")
    Id = np.eye(local_dim, dtype=complex)
    ops = [Id] * n_sites
    ops[site_i] = op_i
    ops[site_j] = op_j
    return kron_all(ops)


def spin_edge_hamiltonian(site_i, site_j, couplings, n_sites, local_dim,
                          normalized=True):
    """XYZ edge  Jx Sx Sx + Jy Sy Sy + Jz Sz Sz  at arbitrary spin."""
    Sx, Sy, Sz = spin_ops(local_dim, normalized=normalized)
    H = np.zeros((local_dim ** n_sites,) * 2, dtype=complex)
    for J, S in zip(couplings, (Sx, Sy, Sz)):
        H += J * embed_two_body(S, S, site_i, site_j, n_sites, local_dim)
    return H


# ---------------------------------------------------------------------
# Unitary log with branch guard (Section 3, large-epsilon work)
# ---------------------------------------------------------------------

def log_unitary(U, branch_guard=True, branch_margin=0.98):
    """
    Principal log of a unitary via eigendecomposition.
    Raises if any eigenphase exceeds branch_margin * pi (guard on).
    """
    w, V = np.linalg.eig(U)
    phases = np.angle(w)
    if branch_guard and np.max(np.abs(phases)) > branch_margin * np.pi:
        raise ValueError(
            f"log branch guard: max |eigenphase| = {np.max(np.abs(phases)):.4f}"
            f" exceeds {branch_margin:.2f} * pi"
        )
    return antihermitian_part(V @ np.diag(1j * phases) @ np.linalg.inv(V))


def protocol_log_eig(hamiltonians, pulse_areas, branch_guard=True):
    """protocol_log with the eig-based branch-guarded logarithm."""
    return log_unitary(protocol_unitary(hamiltonians, pulse_areas),
                       branch_guard=branch_guard)


# ---------------------------------------------------------------------
# Multidegree extraction engine (blueprint Cell 4)
# ---------------------------------------------------------------------

_STENCILS = {
    0: [(0, 1.0)],
    1: [(+1, 0.5), (-1, -0.5)],
    2: [(+1, 1.0), (0, -2.0), (-1, 1.0)],
}


def multidegree_component(hamiltonians, degrees, step, log_fn=None):
    """
    Coefficient of  prod_i s_i^{k_i}  in L(s) = log U_protocol(s).

    degrees : tuple of {0,1,2} per chronological pulse slot.
    Returns the L-layer coefficient (anti-Hermitian); K-layer is i * result.
    """
    if log_fn is None:
        log_fn = protocol_log_eig
    if len(hamiltonians) != len(degrees):
        raise ValueError("hamiltonians and degrees must align")
    for k in degrees:
        if k not in _STENCILS:
            raise ValueError("only degrees 0, 1, 2 are supported here")

    dim = hamiltonians[0].shape[0]
    total = np.zeros((dim, dim), dtype=complex)
    for nodes_weights in product(*[_STENCILS[k] for k in degrees]):
        nodes = [nw[0] for nw in nodes_weights]
        weight = np.prod([nw[1] for nw in nodes_weights])
        areas = [node * step for node in nodes]
        total += weight * log_fn(hamiltonians, areas)

    derivative = total / step ** sum(degrees)
    norm = np.prod([factorial(k) for k in degrees])
    return derivative / norm


def converged_component(hamiltonians, degrees, step, log_fn=None):
    """Run at step and step/2; return (value_at_half, coarse/fine deviation ratio)."""
    coarse = multidegree_component(hamiltonians, degrees, step, log_fn)
    fine = multidegree_component(hamiltonians, degrees, step / 2, log_fn)
    dev = np.linalg.norm(coarse - fine)
    return fine, dev / max(np.linalg.norm(fine), 1e-300)


# ---------------------------------------------------------------------
# Weight decomposition, d-general (blueprint Cell 5)
# ---------------------------------------------------------------------

def _project_trivial_on_site(operator, site, local_dim, n_sites):
    """(I/d)_site  x  Tr_site(O): projector onto 'acts trivially on site'."""
    T = operator.reshape([local_dim] * (2 * n_sites))
    Tr = np.trace(T, axis1=site, axis2=site + n_sites)
    out = np.zeros([local_dim] * (2 * n_sites), dtype=complex)
    idx = [slice(None)] * (2 * n_sites)
    for k in range(local_dim):
        idx[site] = k
        idx[site + n_sites] = k
        out[tuple(idx)] = Tr / local_dim
    return out.reshape(local_dim ** n_sites, local_dim ** n_sites)


def exact_support_component(operator, sites, local_dim, n_sites):
    """Component supported exactly on `sites` (subset-Moebius on the projectors)."""
    P = operator.copy()
    for q in range(n_sites):
        Pq = _project_trivial_on_site(P, q, local_dim, n_sites)
        P = (P - Pq) if q in sites else Pq
    return P


def weight_decomposition(operator, local_dim, n_sites):
    """Dict weight -> squared-norm fraction (exact, any local dimension)."""
    w = {k: 0.0 for k in range(n_sites + 1)}
    for T in product([0, 1], repeat=n_sites):
        sites = {q for q in range(n_sites) if T[q] == 1}
        comp = exact_support_component(operator, sites, local_dim, n_sites)
        w[len(sites)] += float(np.linalg.norm(comp)) ** 2
    total = sum(w.values())
    return {k: v / max(total, 1e-300) for k, v in w.items()}


def pair_ledger(operator, local_dim, n_sites, tol=1e-10):
    """List of (pair, norm) for the exactly-two-local content, pair-resolved."""
    rows = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            comp = exact_support_component(operator, {i, j}, local_dim, n_sites)
            nrm = float(np.linalg.norm(comp))
            if nrm > tol:
                rows.append(((i, j), nrm))
    return rows


# ---------------------------------------------------------------------
# Protocol builders (blueprint Cell 2 derivatives)
# ---------------------------------------------------------------------

def trotter_lists(hamiltonians, eps):
    """First-order Trotter step: one pulse per edge, chronological order."""
    return list(hamiltonians), [eps] * len(hamiltonians)


def strang_lists(H1, H2, eps):
    """Symmetric Strang step  H1/2 -> H2 -> H1/2."""
    return [H1, H2, H1], [eps / 2, eps, eps / 2]


def palindrome_lists(hamiltonians, areas):
    """
    Time-symmetric sequence: H1..H_{n-1} at areas/2, Hn at full area,
    then mirrored. `areas` may carry independent dials for stencils.
    """
    hams = list(hamiltonians[:-1]) + [hamiltonians[-1]] + list(hamiltonians[-2::-1])
    half = [a / 2 for a in areas[:-1]]
    ars = half + [areas[-1]] + half[::-1]
    return hams, ars


def palindrome_log(hamiltonians, areas, log_fn=None):
    if log_fn is None:
        log_fn = protocol_log_eig
    hams, ars = palindrome_lists(hamiltonians, areas)
    return log_fn(hams, ars)


def group_commutator_unitary(H1, H2, H3, eps):
    """
    Twelve-pulse witness  W = U1 V U1^{-1} V^{-1},  V = U2 U3 U2^{-1} U3^{-1},
    built from protocol_unitary blocks (matrix-product notation).
    Leading order:  W ~ exp(+i eps^3 [H1, [H2, H3]]).
    """
    from scipy.linalg import expm
    U = lambda H, s: expm(-1j * s * H)
    V = U(H2, eps) @ U(H3, eps) @ U(H2, -eps) @ U(H3, -eps)
    return U(H1, eps) @ V @ U(H1, -eps) @ np.linalg.inv(V)


# ---------------------------------------------------------------------
# Channel basis and projection (triangle cycle sector)
# ---------------------------------------------------------------------

def channel_basis(H_AB, H_BC, H_AC):
    """Quarter-normalized outer-edge channels (C_AB, C_BC); C_AC = C_AB + C_BC."""
    C_AB = 0.25 * commutator(H_AB, commutator(H_AC, H_BC))
    C_BC = 0.25 * commutator(H_BC, commutator(H_AB, H_AC))
    return C_AB, C_BC


def project_onto_channels(K, channels):
    """Least-squares coordinates of K in the channel basis; returns (coeffs, rel_residual)."""
    basis = np.column_stack([c.reshape(-1) for c in channels])
    coeffs, *_ = np.linalg.lstsq(basis, K.reshape(-1), rcond=None)
    recon = sum(c * ch for c, ch in zip(coeffs, channels))
    rel = np.linalg.norm(K - recon) / max(np.linalg.norm(K), 1e-300)
    return coeffs, rel


def random_xyz_couplings(rng, scale=1.0):
    """Sign-symmetric random XYZ couplings (institutionalizes the octant fix)."""
    v = rng.normal(size=3)
    return scale * v / np.linalg.norm(v)


# ---------------------------------------------------------------------
# Convention self-tests: the Section-0 gate
# ---------------------------------------------------------------------

def run_convention_selftests(verbose=True, rtol=2e-3):
    rng = np.random.default_rng(1)
    H1 = general_edge_hamiltonian(0, 1, rng.uniform(-1, 1, (3, 3)), 3)
    H2 = general_edge_hamiltonian(1, 2, rng.uniform(-1, 1, (3, 3)), 3)
    H3 = general_edge_hamiltonian(0, 2, rng.uniform(-1, 1, (3, 3)), 3)
    C = commutator
    checks = []

    # (1) EVEN anchor: s1 s2 coefficient of L for chronology H1 -> H2 is +1/2 [H1, H2]
    L11 = multidegree_component([H1, H2], (1, 1), 0.004)
    checks.append(("even (1,1) = +1/2 [H1,H2]",
                   np.linalg.norm(L11 - 0.5 * C(H1, H2)) / np.linalg.norm(L11)))

    # (2) ODD anchor: (1,1,1) = i(1/3 C1 - 1/6 C2), reversal-invariant
    L111 = multidegree_component([H1, H2, H3], (1, 1, 1), 0.01)
    tgt = 1j * (C(H1, C(H2, H3)) / 3 - C(H2, C(H1, H3)) / 6)
    checks.append(("odd (1,1,1) = i(C1/3 - C2/6)",
                   np.linalg.norm(L111 - tgt) / np.linalg.norm(L111)))
    L111r = multidegree_component([H3, H2, H1], (1, 1, 1), 0.01)
    checks.append(("odd reversal-invariant", np.linalg.norm(L111r - L111) / np.linalg.norm(L111)))

    # (3) EVEN flips under reversal
    L11r = multidegree_component([H2, H1], (1, 1), 0.004)
    checks.append(("even reversal flip", np.linalg.norm(L11r + L11) / np.linalg.norm(L11)))

    # (4) wedge (2,1) = (i/12)[H1,[H1,H2]]
    L21 = multidegree_component([H1, H2], (2, 1), 0.02)
    checks.append(("(2,1) = (i/12)[H1,[H1,H2]]",
                   np.linalg.norm(L21 - 1j / 12 * C(H1, C(H1, H2))) / np.linalg.norm(L21)))

    # (5) palindrome trilinear = i(1/12 C1 - 1/6 C2)
    Lp = multidegree_component([H1, H2, H3], (1, 1, 1), 0.01,
                               log_fn=lambda h, a: palindrome_log(h, a))
    tgt_p = 1j * (C(H1, C(H2, H3)) / 12 - C(H2, C(H1, H3)) / 6)
    checks.append(("palindrome (1,1,1) = i(C1/12 - C2/6)",
                   np.linalg.norm(Lp - tgt_p) / np.linalg.norm(Lp)))

    # (6) witness leading term
    eps = 0.05
    W = group_commutator_unitary(H1, H2, H3, eps)
    from scipy.linalg import expm
    T = expm(1j * eps ** 3 * C(H1, C(H2, H3)))
    r_id = np.linalg.norm(W - np.eye(8))
    r_tg = np.linalg.norm(W - T)
    checks.append(("witness: ||W-target|| << ||W-1||", r_tg / r_id))

    # (7) weight decomposition sanity
    O = two_body(X, Y, 0, 1, 3)
    w = weight_decomposition(O, 2, 3)
    checks.append(("weight_decomposition sanity", abs(w[2] - 1.0)))

    ok = True
    for name, err in checks:
        passed = err < (0.3 if "witness" in name else rtol)
        ok &= passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {err:.2e}")
    if not ok:
        raise AssertionError("convention self-tests failed")
    return True


if __name__ == "__main__":
    print(CONVENTION_VERSION)
    run_convention_selftests()
