"""
Shared operator-algebra and chronological-protocol utilities
for the Quantum HOI Stage 1 / Stage 2 notebooks.

Frozen conventions
------------------
1. A list [H1, H2, ..., Hn] is in chronological order:
       H1 acts first, Hn acts last.

2. Operators act on states from right to left:
       U = exp(-i s_n H_n) ... exp(-i s_2 H_2) exp(-i s_1 H_1).

3. L = log(U) is returned as an anti-Hermitian matrix.

4. The corresponding Hermitian generator coefficient is K = i L.
"""

from itertools import product

import numpy as np
import pandas as pd
from scipy.linalg import expm, logm


DEFAULT_TOL = 1e-10


# ---------------------------------------------------------------------
# Local operator bases
# ---------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)

X = np.array(
    [[0, 1],
     [1, 0]],
    dtype=complex
)

Y = np.array(
    [[0, -1j],
     [1j, 0]],
    dtype=complex
)

Z = np.array(
    [[1, 0],
     [0, -1]],
    dtype=complex
)

PAULI_BASIS = {
    "I": I2,
    "X": X,
    "Y": Y,
    "Z": Z,
}

PAULI_AXES = {
    "X": X,
    "Y": Y,
    "Z": Z,
}


# ---------------------------------------------------------------------
# Tensor-product embeddings and basic algebra
# ---------------------------------------------------------------------

def kron_all(operators):
    """Kronecker product of a non-empty ordered operator list."""
    if len(operators) == 0:
        raise ValueError("operators must be non-empty")

    result = np.asarray(operators[0], dtype=complex)

    for operator in operators[1:]:
        result = np.kron(result, operator)

    return result


def one_body(operator, site, n_sites):
    """Embed a one-site qubit operator on `site`."""
    operators = [I2] * n_sites
    operators[site] = operator
    return kron_all(operators)


def two_body(operator_i, operator_j, site_i, site_j, n_sites):
    """Embed operator_i ⊗ operator_j on two distinct qubit sites."""
    if site_i == site_j:
        raise ValueError("site_i and site_j must be distinct")

    operators = [I2] * n_sites
    operators[site_i] = operator_i
    operators[site_j] = operator_j
    return kron_all(operators)


def commutator(A, B):
    """Matrix commutator [A, B]."""
    return A @ B - B @ A


def hermitian_part(A):
    """Hermitian part of a square matrix."""
    return 0.5 * (A + A.conj().T)


def antihermitian_part(A):
    """Anti-Hermitian part of a square matrix."""
    return 0.5 * (A - A.conj().T)


def jordan(A, B):
    """Jordan product A o B = (AB + BA)/2."""
    return 0.5 * (A @ B + B @ A)


def jordan_associator(A, B, C):
    """
    (A o B) o C - A o (B o C)
    = 1/4 [B, [A, C]].
    """
    return jordan(jordan(A, B), C) - jordan(A, jordan(B, C))


def three_edge_operator(H_inner_1, H_inner_2, H_outer):
    """
    Algebraic nested word

        1/4 [H_outer, [H_inner_1, H_inner_2]].

    The argument order is a bracketing convention, not a pulse chronology.
    """
    return 0.25 * commutator(
        H_outer,
        commutator(H_inner_1, H_inner_2),
    )


# ---------------------------------------------------------------------
# Pairwise Hamiltonians
# ---------------------------------------------------------------------

def edge_hamiltonian(site_i, site_j, couplings, n_sites):
    """
    Diagonal XYZ edge Hamiltonian

        H_ij = Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j.
    """
    Jx, Jy, Jz = couplings

    return (
        Jx * two_body(X, X, site_i, site_j, n_sites)
        + Jy * two_body(Y, Y, site_i, site_j, n_sites)
        + Jz * two_body(Z, Z, site_i, site_j, n_sites)
    )


def general_edge_hamiltonian(site_i, site_j, couplings, n_sites):
    """
    General real bilinear qubit edge Hamiltonian

        H_ij = sum_{mu,nu in {X,Y,Z}}
               J[mu,nu] sigma_i^mu sigma_j^nu.

    `couplings` must be a 3 x 3 array.
    """
    couplings = np.asarray(couplings)

    if couplings.shape != (3, 3):
        raise ValueError("couplings must have shape (3, 3)")

    axes = [X, Y, Z]
    dimension = 2**n_sites
    H = np.zeros((dimension, dimension), dtype=complex)

    for mu in range(3):
        for nu in range(3):
            H += (
                couplings[mu, nu]
                * two_body(
                    axes[mu],
                    axes[nu],
                    site_i,
                    site_j,
                    n_sites,
                )
            )

    return H


def single_edge_word(
    site_i,
    site_j,
    letter_i,
    letter_j,
    n_sites=3,
):
    """Single Pauli-string edge operator."""
    return two_body(
        PAULI_BASIS[letter_i.upper()],
        PAULI_BASIS[letter_j.upper()],
        site_i,
        site_j,
        n_sites,
    )


def third_pauli(letter_1, letter_2):
    """Return the third Pauli letter, or None when the letters match."""
    letter_1 = letter_1.upper()
    letter_2 = letter_2.upper()

    if letter_1 == letter_2:
        return None

    valid = {"X", "Y", "Z"}

    if letter_1 not in valid or letter_2 not in valid:
        raise ValueError("letters must belong to {'X', 'Y', 'Z'}")

    return (valid - {letter_1, letter_2}).pop()


# ---------------------------------------------------------------------
# Qubit Pauli decomposition
# ---------------------------------------------------------------------

def pauli_decomposition(operator, n_sites, tol=DEFAULT_TOL):
    """
    Decompose a qubit operator into Pauli strings.

    Intended for small qubit systems. The cost scales as 4**n_sites.
    """
    rows = []

    for labels in product("IXYZ", repeat=n_sites):
        P = kron_all([PAULI_BASIS[label] for label in labels])
        coefficient = np.trace(P @ operator) / (2**n_sites)

        if abs(coefficient) > tol:
            rows.append(
                {
                    "pauli_string": "".join(labels),
                    "coefficient_real": float(coefficient.real),
                    "coefficient_imag": float(coefficient.imag),
                    "support_size": sum(label != "I" for label in labels),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "pauli_string",
            "coefficient_real",
            "coefficient_imag",
            "support_size",
        ],
    )


# ---------------------------------------------------------------------
# Chronological pulse protocols
# ---------------------------------------------------------------------

def protocol_unitary(hamiltonians, pulse_areas):
    """
    Construct a pulse protocol from chronological input lists.

    Parameters
    ----------
    hamiltonians : sequence of square arrays
        [H1, H2, ..., Hn], with H1 acting first.
    pulse_areas : sequence of numbers
        [s1, s2, ..., sn].

    Returns
    -------
    U : ndarray
        U = exp(-i s_n H_n) ... exp(-i s_1 H_1).
    """
    if len(hamiltonians) == 0:
        raise ValueError("hamiltonians must be non-empty")

    if len(hamiltonians) != len(pulse_areas):
        raise ValueError(
            "hamiltonians and pulse_areas must have the same length"
        )

    dimension = hamiltonians[0].shape[0]
    U = np.eye(dimension, dtype=complex)

    for H, pulse_area in zip(hamiltonians, pulse_areas):
        if H.shape != (dimension, dimension):
            raise ValueError("all Hamiltonians must have the same shape")

        # Left multiplication implements chronological action.
        U = expm(-1j * pulse_area * H) @ U

    return U


def protocol_log(hamiltonians, pulse_areas):
    """
    Principal matrix logarithm of a chronological pulse protocol.

    The returned matrix is explicitly projected onto its anti-Hermitian part.
    Branch diagnostics are deliberately handled in the Stage 2 notebook.
    """
    U = protocol_unitary(hamiltonians, pulse_areas)
    return antihermitian_part(logm(U))


def protocol_log_2(H1, H2, s1, s2):
    """Two-pulse compatibility wrapper for chronology H1 -> H2."""
    return protocol_log([H1, H2], [s1, s2])


def protocol_log_3(H1, H2, H3, s1, s2, s3):
    """Three-pulse compatibility wrapper for chronology H1 -> H2 -> H3."""
    return protocol_log([H1, H2, H3], [s1, s2, s3])


__all__ = [
    "DEFAULT_TOL",
    "I2",
    "X",
    "Y",
    "Z",
    "PAULI_BASIS",
    "PAULI_AXES",
    "kron_all",
    "one_body",
    "two_body",
    "commutator",
    "hermitian_part",
    "antihermitian_part",
    "jordan",
    "jordan_associator",
    "three_edge_operator",
    "edge_hamiltonian",
    "general_edge_hamiltonian",
    "single_edge_word",
    "third_pauli",
    "pauli_decomposition",
    "protocol_unitary",
    "protocol_log",
    "protocol_log_2",
    "protocol_log_3",
]
