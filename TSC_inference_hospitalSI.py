"""
TSC_inference_hospitalSI.py

SI-adapted endpoint inference for the hospital temporal-network benchmark.

Given the SI governing equation

    F_i(x) = (1 - x_i) * sum_j A_ij x_j,

and a first-order temporal/BCH expansion

    F_eff = F^(0) + epsilon F^(1) + O(epsilon^2),

pairwise SI implies the admissible first-order closure

    F_i^(1)(x) = (1 - x_i) [
        sum_{j != i} alpha_ij x_j
        + sum_{j < k, j,k != i} beta_i;jk x_j x_k
    ].

The estimator therefore uses the known governing equation to build an
SI-adapted dictionary and performs stable column-scaled least squares.  It
never uses the microscopic graph, snapshot chronology, adjacency tensors, or
an exact oracle.

Pipeline
--------
{X0, X_epsilon}
    -> extrapolate F0_data and G1_data
    -> infer persistent SI matrix A0_hat
    -> Q[F0_hat] = 1/2 J_F0_hat F0_hat
    -> F1_target = G1_data - Q[F0_hat]
    -> M2: best pair-only temporal model
    -> M3: pair + triad SI closure

Important distinction
---------------------
M2 is the best pairwise projection of the full temporal correction.  Its pair
coefficients generally need not equal the pair-sector coefficients in M3.

Version: 0.1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import time

import numpy as np
import pandas as pd
from scipy import linalg


__all__ = [
    "HospitalSIInference",
    "HospitalSIResult",
    "TSCInferenceHospitalSI",
]
__version__ = "0.1.0"

ArrayLike = Union[np.ndarray, Sequence[float]]


# ============================================================================
# Utilities
# ============================================================================

def _as_2d_float(X: ArrayLike, *, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_nodes).")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return X


def _as_endpoint_array(
    XF: ArrayLike,
    *,
    n_eps: int,
    n_samples: int,
    n_nodes: int,
    name: str,
) -> np.ndarray:
    XF = np.asarray(XF, dtype=np.float64)
    expected = (n_eps, n_samples, n_nodes)
    if XF.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {XF.shape}.")
    if not np.all(np.isfinite(XF)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return XF


def _relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    denom = max(np.linalg.norm(truth), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(estimate - truth) / denom)


def _rms_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    diff = np.asarray(estimate, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _field_rms(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    return float(np.sqrt(np.mean(X * X)))


def _field_cosine(A: np.ndarray, B: np.ndarray) -> float:
    a = np.asarray(A, dtype=np.float64).ravel()
    b = np.asarray(B, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return np.nan
    return float(np.dot(a, b) / denom)


def _fit_scaled_lstsq(
    Theta: np.ndarray,
    y: np.ndarray,
    *,
    lapack_driver: str = "gelsd",
    rcond: Optional[float] = None,
) -> Dict[str, object]:
    """Column-RMS-scaled least squares; never forms normal equations."""
    Theta = np.asarray(Theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if Theta.ndim != 2:
        raise ValueError("Theta must be two-dimensional.")
    if y.ndim != 1 or y.shape[0] != Theta.shape[0]:
        raise ValueError("y must have shape (Theta.shape[0],).")

    column_rms = np.sqrt(np.mean(Theta * Theta, axis=0))
    if np.any(column_rms <= 0.0):
        bad = np.flatnonzero(column_rms <= 0.0)
        raise ValueError(f"Zero-RMS design columns at indices {bad[:10].tolist()}.")

    Z = Theta / column_rms[None, :]

    t0 = time.perf_counter()
    coef_scaled, residuals, rank, singular_values = linalg.lstsq(
        Z,
        y,
        cond=rcond,
        lapack_driver=lapack_driver,
        check_finite=False,
    )
    runtime_s = time.perf_counter() - t0

    coef = coef_scaled / column_rms
    singular_values = np.asarray(singular_values, dtype=np.float64)

    if singular_values.size:
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        condition_number = float(sigma_max / sigma_min)
    else:
        sigma_max = np.nan
        sigma_min = np.nan
        condition_number = np.nan

    return {
        "coef": np.asarray(coef, dtype=np.float64),
        "rank": int(rank),
        "condition_number": condition_number,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "column_rms": column_rms,
        "runtime_s": float(runtime_s),
        "residuals": residuals,
    }


def _build_si_design(
    X: np.ndarray,
    output_index: int,
    *,
    include_triads: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Build pair-only or pair+triad SI closure design for one output."""
    X = np.asarray(X, dtype=np.float64)
    _, n_nodes = X.shape

    if not (0 <= output_index < n_nodes):
        raise ValueError("output_index out of range.")

    others = np.concatenate(
        (
            np.arange(0, output_index, dtype=np.int32),
            np.arange(output_index + 1, n_nodes, dtype=np.int32),
        )
    )

    gate = 1.0 - X[:, output_index]
    Xo = X[:, others]
    pair_block = gate[:, None] * Xo

    if not include_triads:
        return pair_block, others, None, None

    j_local, k_local = np.triu_indices(len(others), k=1)
    triad_j = others[j_local]
    triad_k = others[k_local]

    triad_block = gate[:, None] * Xo[:, j_local] * Xo[:, k_local]
    Theta = np.concatenate((pair_block, triad_block), axis=1)
    return Theta, others, triad_j, triad_k


def _evaluate_F0_from_matrix(X: np.ndarray, A0: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    pressure = X @ A0.T
    return (1.0 - X) * pressure


def _Q_from_F0_matrix(X: np.ndarray, A0: np.ndarray) -> np.ndarray:
    """Q[F0] = 1/2 J_F0 F0 for F0_i=(1-x_i)(A0 x)_i."""
    X = np.asarray(X, dtype=np.float64)
    pressure = X @ A0.T
    F0 = (1.0 - X) * pressure
    AF = F0 @ A0.T
    JF = (1.0 - X) * AF - pressure * F0
    return 0.5 * JF


# ============================================================================
# Result object
# ============================================================================

@dataclass
class HospitalSIResult:
    epsilon_used: np.ndarray

    F0_data_train: np.ndarray
    G1_data_train: np.ndarray
    F1_target_train: np.ndarray

    F0_data_test: Optional[np.ndarray]
    G1_data_test: Optional[np.ndarray]
    F1_target_test: Optional[np.ndarray]

    A0: np.ndarray
    alpha_pair_only: np.ndarray
    alpha_pair_triad: np.ndarray
    beta_pair_triad: np.ndarray

    metrics: Dict[str, float]
    output_diagnostics: pd.DataFrame
    config: Dict[str, object]

    @property
    def n_nodes(self) -> int:
        return int(self.A0.shape[0])

    def predict_F0(self, X: ArrayLike) -> np.ndarray:
        X = _as_2d_float(X, name="X")
        if X.shape[1] != self.n_nodes:
            raise ValueError("X has the wrong number of nodes.")
        return _evaluate_F0_from_matrix(X, self.A0)

    def predict_Q(self, X: ArrayLike) -> np.ndarray:
        X = _as_2d_float(X, name="X")
        if X.shape[1] != self.n_nodes:
            raise ValueError("X has the wrong number of nodes.")
        return _Q_from_F0_matrix(X, self.A0)

    def predict_F1(self, X: ArrayLike, *, model: str = "pair_triad") -> np.ndarray:
        X = _as_2d_float(X, name="X")
        if X.shape[1] != self.n_nodes:
            raise ValueError("X has the wrong number of nodes.")

        model = str(model).lower()
        if model not in {"pair_only", "pair_triad"}:
            raise ValueError("model must be 'pair_only' or 'pair_triad'.")

        n_samples, n_nodes = X.shape
        Y = np.zeros_like(X)
        alpha = self.alpha_pair_only if model == "pair_only" else self.alpha_pair_triad

        for i in range(n_nodes):
            gate = 1.0 - X[:, i]
            Y[:, i] = gate * (X @ alpha[i])

            if model == "pair_triad":
                others = np.concatenate(
                    (
                        np.arange(0, i, dtype=np.int32),
                        np.arange(i + 1, n_nodes, dtype=np.int32),
                    )
                )
                j_local, k_local = np.triu_indices(len(others), k=1)
                j_idx = others[j_local]
                k_idx = others[k_local]
                beta = self.beta_pair_triad[i, j_idx, k_idx]
                Y[:, i] += gate * ((X[:, j_idx] * X[:, k_idx]) @ beta)

        return Y

    def predict_G1(self, X: ArrayLike, *, model: str = "pair_triad") -> np.ndarray:
        X = _as_2d_float(X, name="X")
        return self.predict_Q(X) + self.predict_F1(X, model=model)

    def persistent_pair_table(self, *, threshold: float = 0.0) -> pd.DataFrame:
        rows = []
        N = self.n_nodes
        threshold = float(threshold)

        for i in range(N):
            for j in range(i + 1, N):
                cij = float(self.A0[i, j])
                cji = float(self.A0[j, i])
                strength = float(np.hypot(cij, cji))
                if strength <= threshold:
                    continue
                rows.append(
                    {
                        "support": (i + 1, j + 1),
                        "interaction_order": 2,
                        "coef_i_from_j": cij,
                        "coef_j_from_i": cji,
                        "coefficient_norm": strength,
                    }
                )
        return pd.DataFrame(rows)

    def temporal_pair_table(
        self,
        *,
        model: str = "pair_triad",
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        model = str(model).lower()
        if model == "pair_only":
            alpha = self.alpha_pair_only
        elif model == "pair_triad":
            alpha = self.alpha_pair_triad
        else:
            raise ValueError("model must be 'pair_only' or 'pair_triad'.")

        rows = []
        N = self.n_nodes
        threshold = float(threshold)

        for i in range(N):
            for j in range(i + 1, N):
                cij = float(alpha[i, j])
                cji = float(alpha[j, i])
                strength = float(np.hypot(cij, cji))
                if strength <= threshold:
                    continue
                rows.append(
                    {
                        "support": (i + 1, j + 1),
                        "interaction_order": 2,
                        "coef_i_from_j": cij,
                        "coef_j_from_i": cji,
                        "coefficient_norm": strength,
                    }
                )
        return pd.DataFrame(rows)

    def temporal_triad_table(self, *, threshold: float = 0.0) -> pd.DataFrame:
        rows = []
        N = self.n_nodes
        threshold = float(threshold)
        beta = self.beta_pair_triad

        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    bi = float(beta[i, j, k])
                    bj = float(beta[j, i, k])
                    bk = float(beta[k, i, j])
                    strength = float(np.sqrt(bi * bi + bj * bj + bk * bk))
                    if strength <= threshold:
                        continue
                    rows.append(
                        {
                            "support": (i + 1, j + 1, k + 1),
                            "interaction_order": 3,
                            "beta_i_jk": bi,
                            "beta_j_ik": bj,
                            "beta_k_ij": bk,
                            "coefficient_norm": strength,
                        }
                    )
        return pd.DataFrame(rows)

    def structure_table(
        self,
        *,
        temporal_order: Optional[int] = None,
        threshold: float = 0.0,
        temporal_model: str = "pair_triad",
    ) -> pd.DataFrame:
        if temporal_order not in {None, 0, 1}:
            raise ValueError("temporal_order must be None, 0, or 1.")

        frames = []

        if temporal_order in {None, 0}:
            f0 = self.persistent_pair_table(threshold=threshold)
            if len(f0):
                f0.insert(0, "temporal_order", 0)
                f0.insert(1, "sector", "persistent_pair")
                frames.append(f0)

        if temporal_order in {None, 1}:
            f1p = self.temporal_pair_table(
                model=temporal_model,
                threshold=threshold,
            )
            if len(f1p):
                f1p.insert(0, "temporal_order", 1)
                f1p.insert(1, "sector", f"temporal_pair_{temporal_model}")
                frames.append(f1p)

            if temporal_model == "pair_triad":
                f1t = self.temporal_triad_table(threshold=threshold)
                if len(f1t):
                    f1t.insert(0, "temporal_order", 1)
                    f1t.insert(1, "sector", "temporal_triad")
                    frames.append(f1t)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "temporal_order",
                    "sector",
                    "support",
                    "interaction_order",
                    "coefficient_norm",
                ]
            )

        return pd.concat(frames, ignore_index=True, sort=False)

    def summary(self) -> pd.DataFrame:
        keys = [
            ("F0 train relative error", "F0_train_rel_error"),
            ("F0 test relative error", "F0_test_rel_error"),
            ("M2 train relative error", "M2_train_rel_error"),
            ("M2 test relative error", "M2_test_rel_error"),
            ("M3 train relative error", "M3_train_rel_error"),
            ("M3 test relative error", "M3_test_rel_error"),
            ("M2/M3 held-out improvement", "M2_over_M3_test_improvement"),
            ("M2 field RMS", "M2_field_rms_train"),
            ("M3 pair field RMS", "M3_pair_field_rms_train"),
            ("M3 triad field RMS", "M3_triad_field_rms_train"),
            ("M3 pair-triad field cosine", "M3_pair_triad_cosine_train"),
            ("Total runtime [s]", "total_runtime_s"),
        ]
        return pd.DataFrame(
            [
                {"metric": label, "value": self.metrics[key]}
                for label, key in keys
                if key in self.metrics
            ]
        )


# ============================================================================
# Estimator
# ============================================================================

class HospitalSIInference:
    """Endpoint-only SI-adapted first-order TSC inference."""

    def __init__(
        self,
        *,
        extrapolation_points: int = 4,
        extrapolation_order: int = 3,
        fit_pair_only: bool = True,
        fit_pair_triad: bool = True,
        lapack_driver: str = "gelsd",
        rcond: Optional[float] = None,
        verbose: bool = True,
    ):
        self.extrapolation_points = int(extrapolation_points)
        self.extrapolation_order = int(extrapolation_order)
        self.fit_pair_only = bool(fit_pair_only)
        self.fit_pair_triad = bool(fit_pair_triad)
        self.lapack_driver = str(lapack_driver)
        self.rcond = rcond
        self.verbose = bool(verbose)

        if self.extrapolation_points < 2:
            raise ValueError("extrapolation_points must be at least 2.")
        if self.extrapolation_order < 1:
            raise ValueError("extrapolation_order must be at least 1.")
        if self.extrapolation_order >= self.extrapolation_points:
            raise ValueError(
                "extrapolation_order must be smaller than extrapolation_points."
            )
        if not (self.fit_pair_only or self.fit_pair_triad):
            raise ValueError("At least one temporal model must be fitted.")

    def _extrapolate(
        self,
        X0: np.ndarray,
        XF: np.ndarray,
        epsilon_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate F0 and G1 from forward endpoint secants."""
        order = np.argsort(epsilon_values)
        chosen = order[: self.extrapolation_points]

        eps = epsilon_values[chosen]
        XF_small = XF[chosen]
        secants = (XF_small - X0[None, :, :]) / eps[:, None, None]

        eps_scale = float(np.max(eps))
        z = eps / eps_scale
        V = np.column_stack(
            [z ** power for power in range(self.extrapolation_order + 1)]
        )

        coeff, _, _, _ = linalg.lstsq(
            V,
            secants.reshape(len(eps), -1),
            cond=None,
            lapack_driver="gelsd",
            check_finite=False,
        )
        coeff = coeff.reshape(
            self.extrapolation_order + 1,
            X0.shape[0],
            X0.shape[1],
        )

        F0_data = coeff[0]
        G1_data = coeff[1] / eps_scale
        return F0_data, G1_data, eps.copy()

    def fit(
        self,
        X0_train: ArrayLike,
        XF_train: ArrayLike,
        epsilon_values: ArrayLike,
        *,
        X0_test: Optional[ArrayLike] = None,
        XF_test: Optional[ArrayLike] = None,
    ) -> HospitalSIResult:
        t_total = time.perf_counter()

        X0_train = _as_2d_float(X0_train, name="X0_train")
        n_train, n_nodes = X0_train.shape

        epsilon_values = np.asarray(epsilon_values, dtype=np.float64)
        if epsilon_values.ndim != 1:
            raise ValueError("epsilon_values must be one-dimensional.")
        if len(epsilon_values) < self.extrapolation_points:
            raise ValueError("Not enough epsilon values for extrapolation.")
        if np.any(~np.isfinite(epsilon_values)) or np.any(epsilon_values <= 0):
            raise ValueError("epsilon_values must be finite and strictly positive.")
        if len(np.unique(epsilon_values)) != len(epsilon_values):
            raise ValueError("epsilon_values must be distinct.")

        XF_train = _as_endpoint_array(
            XF_train,
            n_eps=len(epsilon_values),
            n_samples=n_train,
            n_nodes=n_nodes,
            name="XF_train",
        )

        has_test = (X0_test is not None) or (XF_test is not None)
        if has_test and (X0_test is None or XF_test is None):
            raise ValueError("Provide both X0_test and XF_test, or neither.")

        if has_test:
            X0_test = _as_2d_float(X0_test, name="X0_test")
            if X0_test.shape[1] != n_nodes:
                raise ValueError("X0_test has the wrong number of nodes.")
            XF_test = _as_endpoint_array(
                XF_test,
                n_eps=len(epsilon_values),
                n_samples=X0_test.shape[0],
                n_nodes=n_nodes,
                name="XF_test",
            )
        else:
            X0_test = None
            XF_test = None

        if np.min(X0_train) < -1e-8 or np.max(X0_train) > 1.0 + 1e-8:
            raise ValueError(
                "HospitalSIInference expects SI probability-like states in [0,1]."
            )

        pair_features = n_nodes - 1
        triad_features = (n_nodes - 1) * (n_nodes - 2) // 2
        full_features = pair_features + triad_features

        if self.fit_pair_triad and n_train < full_features:
            raise ValueError(
                "Full SI closure is underdetermined for direct OLS: "
                f"n_train={n_train}, features/output={full_features}."
            )

        if self.verbose:
            print("Hospital SI first-order TSC inference")
            print("-------------------------------------")
            print(f"N = {n_nodes}")
            print(f"train states = {n_train}")
            if has_test:
                print(f"test states = {X0_test.shape[0]}")
            print(
                "SI-adapted F1 dictionary/output = "
                f"{pair_features} pair + {triad_features} triad = {full_features}"
            )
            print()

        # ----------------------------------------------------------------
        # 1. Endpoint extrapolation
        # ----------------------------------------------------------------
        if self.verbose:
            print("Extrapolating F^(0) and G^(1) from endpoints...")

        F0_data_train, G1_data_train, eps_used = self._extrapolate(
            X0_train, XF_train, epsilon_values
        )

        if has_test:
            F0_data_test, G1_data_test, eps_used_test = self._extrapolate(
                X0_test, XF_test, epsilon_values
            )
            if not np.array_equal(eps_used, eps_used_test):
                raise RuntimeError("Train/test extrapolation used different epsilons.")
        else:
            F0_data_test = None
            G1_data_test = None

        if self.verbose:
            print(f"  epsilon used = {eps_used}")

        # ----------------------------------------------------------------
        # 2. Persistent F0 sector
        # ----------------------------------------------------------------
        if self.verbose:
            print("Inferring persistent SI coefficient matrix A^(0)...")

        A0 = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        F0_hat_train = np.zeros_like(F0_data_train)
        F0_hat_test = np.zeros_like(F0_data_test) if has_test else None
        diagnostic_rows = []

        for i in range(n_nodes):
            Theta_pair_train, others, _, _ = _build_si_design(
                X0_train, i, include_triads=False
            )
            fit0 = _fit_scaled_lstsq(
                Theta_pair_train,
                F0_data_train[:, i],
                lapack_driver=self.lapack_driver,
                rcond=self.rcond,
            )
            A0[i, others] = fit0["coef"]
            F0_hat_train[:, i] = Theta_pair_train @ fit0["coef"]

            if has_test:
                Theta_pair_test, _, _, _ = _build_si_design(
                    X0_test, i, include_triads=False
                )
                F0_hat_test[:, i] = Theta_pair_test @ fit0["coef"]

            diagnostic_rows.append(
                {
                    "output_node": i + 1,
                    "stage": "F0",
                    "model": "pair",
                    "n_features": pair_features,
                    "rank": fit0["rank"],
                    "condition_number": fit0["condition_number"],
                    "runtime_s": fit0["runtime_s"],
                }
            )

        # ----------------------------------------------------------------
        # 3. Finite-flow correction and F1 target
        # ----------------------------------------------------------------
        if self.verbose:
            print("Constructing Q[F^(0)_hat] and F^(1) target...")

        Q_hat_train = _Q_from_F0_matrix(X0_train, A0)
        F1_target_train = G1_data_train - Q_hat_train

        if has_test:
            Q_hat_test = _Q_from_F0_matrix(X0_test, A0)
            F1_target_test = G1_data_test - Q_hat_test
        else:
            F1_target_test = None

        # ----------------------------------------------------------------
        # 4. Nested temporal models M2 / M3
        # ----------------------------------------------------------------
        alpha_pair_only = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        alpha_pair_triad = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        beta_pair_triad = np.zeros((n_nodes, n_nodes, n_nodes), dtype=np.float64)

        M2_hat_train = np.zeros_like(F1_target_train) if self.fit_pair_only else None
        M3_hat_train = np.zeros_like(F1_target_train) if self.fit_pair_triad else None
        M2_hat_test = (
            np.zeros_like(F1_target_test) if has_test and self.fit_pair_only else None
        )
        M3_hat_test = (
            np.zeros_like(F1_target_test) if has_test and self.fit_pair_triad else None
        )

        if self.verbose:
            print("Fitting temporal closure output by output...")

        for i in range(n_nodes):
            t_output = time.perf_counter()

            Theta_full_train, others, triad_j, triad_k = _build_si_design(
                X0_train, i, include_triads=True
            )
            Theta_pair_train = Theta_full_train[:, :pair_features]

            if has_test:
                Theta_full_test, _, triad_j_test, triad_k_test = _build_si_design(
                    X0_test, i, include_triads=True
                )
                if not (
                    np.array_equal(triad_j, triad_j_test)
                    and np.array_equal(triad_k, triad_k_test)
                ):
                    raise RuntimeError("Train/test triad ordering mismatch.")
                Theta_pair_test = Theta_full_test[:, :pair_features]
            else:
                Theta_full_test = None
                Theta_pair_test = None

            y_train = F1_target_train[:, i]

            if self.fit_pair_only:
                fit2 = _fit_scaled_lstsq(
                    Theta_pair_train,
                    y_train,
                    lapack_driver=self.lapack_driver,
                    rcond=self.rcond,
                )
                alpha_pair_only[i, others] = fit2["coef"]
                M2_hat_train[:, i] = Theta_pair_train @ fit2["coef"]
                if has_test:
                    M2_hat_test[:, i] = Theta_pair_test @ fit2["coef"]

                diagnostic_rows.append(
                    {
                        "output_node": i + 1,
                        "stage": "F1",
                        "model": "M2_pair_only",
                        "n_features": pair_features,
                        "rank": fit2["rank"],
                        "condition_number": fit2["condition_number"],
                        "runtime_s": fit2["runtime_s"],
                    }
                )

            if self.fit_pair_triad:
                fit3 = _fit_scaled_lstsq(
                    Theta_full_train,
                    y_train,
                    lapack_driver=self.lapack_driver,
                    rcond=self.rcond,
                )
                coef3 = fit3["coef"]
                alpha_pair_triad[i, others] = coef3[:pair_features]

                beta_values = coef3[pair_features:]
                beta_pair_triad[i, triad_j, triad_k] = beta_values
                beta_pair_triad[i, triad_k, triad_j] = beta_values

                M3_hat_train[:, i] = Theta_full_train @ coef3
                if has_test:
                    M3_hat_test[:, i] = Theta_full_test @ coef3

                diagnostic_rows.append(
                    {
                        "output_node": i + 1,
                        "stage": "F1",
                        "model": "M3_pair_triad",
                        "n_features": full_features,
                        "rank": fit3["rank"],
                        "condition_number": fit3["condition_number"],
                        "runtime_s": fit3["runtime_s"],
                    }
                )

            if self.verbose:
                elapsed = time.perf_counter() - t_output
                msg = f"  [{i+1:02d}/{n_nodes:02d}] output {i+1:02d}"
                if self.fit_pair_triad:
                    msg += (
                        f" | M3 rank={fit3['rank']}/{full_features}"
                        f" | kappa={fit3['condition_number']:.3e}"
                    )
                msg += f" | elapsed={elapsed:.2f}s"
                print(msg)

        # ----------------------------------------------------------------
        # 5. Metrics
        # ----------------------------------------------------------------
        metrics: Dict[str, float] = {}

        metrics["F0_train_rel_error"] = _relative_error(F0_hat_train, F0_data_train)
        metrics["F0_train_RMS_error"] = _rms_error(F0_hat_train, F0_data_train)
        if has_test:
            metrics["F0_test_rel_error"] = _relative_error(F0_hat_test, F0_data_test)
            metrics["F0_test_RMS_error"] = _rms_error(F0_hat_test, F0_data_test)

        metrics["F1_target_RMS_train"] = _field_rms(F1_target_train)
        if has_test:
            metrics["F1_target_RMS_test"] = _field_rms(F1_target_test)

        if self.fit_pair_only:
            metrics["M2_train_rel_error"] = _relative_error(
                M2_hat_train, F1_target_train
            )
            metrics["M2_train_RMS_error"] = _rms_error(M2_hat_train, F1_target_train)
            metrics["M2_field_rms_train"] = _field_rms(M2_hat_train)
            if has_test:
                metrics["M2_test_rel_error"] = _relative_error(
                    M2_hat_test, F1_target_test
                )
                metrics["M2_test_RMS_error"] = _rms_error(
                    M2_hat_test, F1_target_test
                )

        if self.fit_pair_triad:
            metrics["M3_train_rel_error"] = _relative_error(
                M3_hat_train, F1_target_train
            )
            metrics["M3_train_RMS_error"] = _rms_error(M3_hat_train, F1_target_train)
            if has_test:
                metrics["M3_test_rel_error"] = _relative_error(
                    M3_hat_test, F1_target_test
                )
                metrics["M3_test_RMS_error"] = _rms_error(
                    M3_hat_test, F1_target_test
                )

            pair_train = np.zeros_like(F1_target_train)
            triad_train = np.zeros_like(F1_target_train)
            for i in range(n_nodes):
                gate = 1.0 - X0_train[:, i]
                pair_train[:, i] = gate * (X0_train @ alpha_pair_triad[i])

                others = np.concatenate(
                    (
                        np.arange(0, i, dtype=np.int32),
                        np.arange(i + 1, n_nodes, dtype=np.int32),
                    )
                )
                j_local, k_local = np.triu_indices(len(others), k=1)
                j_idx = others[j_local]
                k_idx = others[k_local]
                b = beta_pair_triad[i, j_idx, k_idx]
                triad_train[:, i] = gate * (
                    (X0_train[:, j_idx] * X0_train[:, k_idx]) @ b
                )

            metrics["M3_pair_field_rms_train"] = _field_rms(pair_train)
            metrics["M3_triad_field_rms_train"] = _field_rms(triad_train)
            metrics["M3_total_field_rms_train"] = _field_rms(pair_train + triad_train)
            metrics["M3_pair_triad_cosine_train"] = _field_cosine(
                pair_train, triad_train
            )

        if has_test and self.fit_pair_only and self.fit_pair_triad:
            denom = max(metrics["M3_test_rel_error"], np.finfo(np.float64).tiny)
            metrics["M2_over_M3_test_improvement"] = (
                metrics["M2_test_rel_error"] / denom
            )

        metrics["total_runtime_s"] = float(time.perf_counter() - t_total)

        config = {
            "version": __version__,
            "n_nodes": n_nodes,
            "n_train": n_train,
            "n_test": int(X0_test.shape[0]) if has_test else 0,
            "extrapolation_points": self.extrapolation_points,
            "extrapolation_order": self.extrapolation_order,
            "epsilon_used": eps_used.copy(),
            "fit_pair_only": self.fit_pair_only,
            "fit_pair_triad": self.fit_pair_triad,
            "pair_features_per_output": pair_features,
            "triad_features_per_output": triad_features,
            "full_features_per_output": full_features,
            "lapack_driver": self.lapack_driver,
            "rcond": self.rcond,
        }

        result = HospitalSIResult(
            epsilon_used=eps_used,
            F0_data_train=F0_data_train,
            G1_data_train=G1_data_train,
            F1_target_train=F1_target_train,
            F0_data_test=F0_data_test,
            G1_data_test=G1_data_test,
            F1_target_test=F1_target_test,
            A0=A0,
            alpha_pair_only=alpha_pair_only,
            alpha_pair_triad=alpha_pair_triad,
            beta_pair_triad=beta_pair_triad,
            metrics=metrics,
            output_diagnostics=pd.DataFrame(diagnostic_rows),
            config=config,
        )

        if self.verbose:
            print()
            print("Inference complete.")
            print(f"  F0 train rel error = {metrics['F0_train_rel_error']:.3e}")
            if has_test:
                print(f"  F0 test rel error  = {metrics['F0_test_rel_error']:.3e}")
            if self.fit_pair_only:
                print(f"  M2 train rel error = {metrics['M2_train_rel_error']:.3e}")
                if has_test:
                    print(f"  M2 test rel error  = {metrics['M2_test_rel_error']:.3e}")
            if self.fit_pair_triad:
                print(f"  M3 train rel error = {metrics['M3_train_rel_error']:.3e}")
                if has_test:
                    print(f"  M3 test rel error  = {metrics['M3_test_rel_error']:.3e}")
            if "M2_over_M3_test_improvement" in metrics:
                print(
                    "  held-out M2/M3 improvement = "
                    f"{metrics['M2_over_M3_test_improvement']:.3e}"
                )
            print(f"  total runtime [s] = {metrics['total_runtime_s']:.2f}")

        return result


TSCInferenceHospitalSI = HospitalSIInference
