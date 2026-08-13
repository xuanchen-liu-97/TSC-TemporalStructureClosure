"""
TSC_AGLASSO.py
==============

Forward-only Temporal Structural Closure (TSC) inference.

Internal implementation: v3
- condition-number-adaptive Ridge pilot for Adaptive Group LASSO weights
- KKT-certified working-set optimization with support reactivation

The public class/module names intentionally remain version-free.

Public modelling hyperparameters
--------------------------------
max_interaction_order
max_polynomial_degree
temporal_order

Validated native input
----------------------
X0 : (M, N)
XF : (L, M, N) or {epsilon: (M, N)}
eps: (L,)

Each row of X0 is paired with the corresponding row of XF at every
resolution epsilon. The learner has no access to microscopic graph,
snapshots, edge weights, or oracle supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from itertools import combinations_with_replacement
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union
import time

import numpy as np
import pandas as pd

Array = np.ndarray
Support = Tuple[int, ...]


@dataclass
class GeneratorFitResult:
    coefficients_scaled: Array
    coefficients_raw: Array
    selected_supports: Tuple[Support, ...]
    train_relative_error: float
    test_relative_error: Optional[float]
    lambda_selected: float
    path_ledger: pd.DataFrame
    group_counts_by_size: Dict[int, int]
    all_solver_runs_converged: bool
    max_solver_iterations: int
    pilot_method: str = "adaptive-ridge"
    ridge_alpha_min: float = np.nan
    ridge_alpha_max: float = np.nan
    kkt_certified: bool = False
    total_kkt_reactivations: int = 0
    max_final_kkt_ratio: float = np.nan

    @property
    def max_interaction_order(self) -> int:
        return max((len(s) for s in self.selected_supports), default=0)


@dataclass
class TSCResult:
    F0: GeneratorFitResult
    F1: GeneratorFitResult
    extrapolation_eps: Array
    A_train: Array
    G1_train: Array
    Q_train: Array
    F1_target_train: Array
    A_test: Optional[Array] = None
    G1_test: Optional[Array] = None
    Q_test: Optional[Array] = None
    F1_target_test: Optional[Array] = None
    diagnostics: Optional[Dict[str, float]] = None
    _library: Optional["StructuralPolynomialLibrary"] = None

    @property
    def k_star(self) -> int:
        return max(
            self.F0.max_interaction_order,
            self.F1.max_interaction_order,
        )

    def _sector_field(
        self,
        X: Array,
        B_raw: Array,
        support: Support,
        Theta_raw: Optional[Array] = None,
    ) -> Array:
        """Evaluate one minimal-support sector of a learned generator."""
        if self._library is None:
            raise RuntimeError("Structural library is unavailable in this result.")
        if support not in self._library.structural_groups:
            raise KeyError(f"Unknown structural support {support}.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._library.n_nodes:
            raise ValueError(
                f"X must have shape (n_samples, {self._library.n_nodes})."
            )

        if Theta_raw is None:
            Theta_raw = self._library.evaluate_raw(X)

        n_nodes = self._library.n_nodes
        indices = self._library.structural_groups[support]
        field = np.zeros((X.shape[0], n_nodes), dtype=float)

        outputs = indices % n_nodes
        features = indices // n_nodes
        for i in np.unique(outputs):
            feat_i = features[outputs == i]
            if len(feat_i):
                field[:, i] = Theta_raw[:, feat_i] @ B_raw[feat_i, i]

        return field

    def beta_S(self, support: Support, temporal_order: int = 0) -> Array:
        """
        Return the raw polynomial coefficient vector beta_S^(r) for one
        minimal structural support S.

        Parameters
        ----------
        support : tuple of 1-based node labels, e.g. (1, 2, 3).
        temporal_order : 0 for F^(0), 1 for F^(1).

        Notes
        -----
        The returned vector contains exactly the coefficient directions whose
        minimal structural support equals S. It is not a scalar coupling.
        """
        if self._library is None:
            raise RuntimeError("Structural library is unavailable in this result.")

        support = tuple(sorted(int(i) for i in support))
        if support not in self._library.structural_groups:
            raise KeyError(f"Unknown structural support {support}.")

        if temporal_order == 0:
            B_raw = self.F0.coefficients_raw
        elif temporal_order == 1:
            B_raw = self.F1.coefficients_raw
        else:
            raise ValueError("temporal_order must be 0 or 1.")

        idx = self._library.structural_groups[support]
        return B_raw.ravel(order="C")[idx].copy()

    def sector_coefficients(
        self,
        support: Support,
        temporal_order: int = 0,
        *,
        nonzero_only: bool = False,
        zero_tol: float = 0.0,
    ) -> pd.DataFrame:
        """
        Return beta_S^(r) with its output-node and polynomial-basis metadata.

        This is the readable/detail-level representation of beta_S. The
        coefficients are the raw (unscaled) polynomial coefficients.
        """
        if self._library is None:
            raise RuntimeError("Structural library is unavailable in this result.")

        support = tuple(sorted(int(i) for i in support))
        if support not in self._library.structural_groups:
            raise KeyError(f"Unknown structural support {support}.")

        if temporal_order == 0:
            B_raw = self.F0.coefficients_raw
        elif temporal_order == 1:
            B_raw = self.F1.coefficients_raw
        else:
            raise ValueError("temporal_order must be 0 or 1.")

        rows = []
        n_nodes = self._library.n_nodes
        for flat in self._library.structural_groups[support]:
            feature = int(flat // n_nodes)
            output = int(flat % n_nodes)
            coefficient = float(B_raw[feature, output])

            if nonzero_only and abs(coefficient) <= float(zero_tol):
                continue

            exponent = self._library.exponents[feature]
            factors = []
            for j, power in enumerate(exponent, start=1):
                if power == 1:
                    factors.append(f"x{j}")
                elif power > 1:
                    factors.append(f"x{j}^{power}")
            monomial = "*".join(factors) if factors else "1"

            rows.append({
                "output": output + 1,
                "monomial": monomial,
                "coefficient": coefficient,
            })

        return pd.DataFrame(rows, columns=["output", "monomial", "coefficient"])

    def structure_table(self) -> pd.DataFrame:
        """
        Return the inferred structural-closure table.

        Columns
        -------
        order   : cardinality |S| of the minimal structural support.
        support : node tuple S (1-based node labels).
        F^(0)   : True iff S is selected in the persistent/zeroth-order sector.
        F^(1)   : True iff S is selected in the first temporal-order sector.
        origin  : persistent-only, temporal-only, or mixed.

        Structural existence is taken directly from the Adaptive Group LASSO
        selected-support sets. No unvalidated scalar weight W_S is introduced
        here. Full raw coefficient information beta_S remains available through
        beta_S() and sector_coefficients().
        """
        supports0 = set(self.F0.selected_supports)
        supports1 = set(self.F1.selected_supports)
        supports = sorted(supports0 | supports1, key=lambda s: (len(s), s))

        rows = []
        for support in supports:
            in0 = support in supports0
            in1 = support in supports1

            if in0 and in1:
                origin = "mixed"
            elif in0:
                origin = "persistent-only"
            else:
                origin = "temporal-only"

            rows.append({
                "order": len(support),
                "support": support,
                "F^(0)": bool(in0),
                "F^(1)": bool(in1),
                "origin": origin,
            })

        return pd.DataFrame(
            rows,
            columns=["order", "support", "F^(0)", "F^(1)", "origin"],
        )

    def result_table(self) -> pd.DataFrame:
        """Alias for structure_table()."""
        return self.structure_table()

    def summary(self, show_supports: bool = False) -> None:
        print("TSC inference complete.")
        print()
        print("F^(0):")
        print(f"  active groups = {len(self.F0.selected_supports)}")
        print(f"  groups by support size = {self.F0.group_counts_by_size}")
        print(f"  max interaction order = {self.F0.max_interaction_order}")
        print(f"  pilot estimator = {self.F0.pilot_method}")
        print(
            "  adaptive ridge alpha range = "
            f"[{self.F0.ridge_alpha_min:.3e}, {self.F0.ridge_alpha_max:.3e}]"
        )
        print(
            f"  KKT certified = {self.F0.kkt_certified} "
            f"| reactivations = {self.F0.total_kkt_reactivations}"
        )
        print(f"  train relative field error = {self.F0.train_relative_error:.3e}")
        if self.F0.test_relative_error is not None:
            print(f"  test relative field error  = {self.F0.test_relative_error:.3e}")

        print()
        print("F^(1):")
        print(f"  active groups = {len(self.F1.selected_supports)}")
        print(f"  groups by support size = {self.F1.group_counts_by_size}")
        print(f"  max interaction order = {self.F1.max_interaction_order}")
        print(f"  pilot estimator = {self.F1.pilot_method}")
        print(
            "  adaptive ridge alpha range = "
            f"[{self.F1.ridge_alpha_min:.3e}, {self.F1.ridge_alpha_max:.3e}]"
        )
        print(
            f"  KKT certified = {self.F1.kkt_certified} "
            f"| reactivations = {self.F1.total_kkt_reactivations}"
        )
        print(f"  train relative field error = {self.F1.train_relative_error:.3e}")
        if self.F1.test_relative_error is not None:
            print(f"  test relative field error  = {self.F1.test_relative_error:.3e}")

        print()
        print(f"Effective closure order k_star = {self.k_star}")

        if self.diagnostics:
            print()
            print("Data-level diagnostics:")
            for key, value in self.diagnostics.items():
                print(f"  {key} = {value:.6e}")

        if show_supports:
            print()
            print("Selected F^(0) supports:")
            for s in self.F0.selected_supports:
                print(" ", s)
            print()
            print("Selected F^(1) supports:")
            for s in self.F1.selected_supports:
                print(" ", s)


class StructuralPolynomialLibrary:
    def __init__(
        self,
        n_nodes: int,
        max_polynomial_degree: int,
        max_interaction_order: int,
    ):
        self.n_nodes = int(n_nodes)
        self.max_polynomial_degree = int(max_polynomial_degree)
        self.max_interaction_order = int(max_interaction_order)

        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive.")
        if self.max_polynomial_degree < 1:
            raise ValueError("max_polynomial_degree must be >= 1.")
        if self.max_interaction_order < 1:
            raise ValueError("max_interaction_order must be >= 1.")

        self.exponents = self._generate_exponents()
        self.n_features = len(self.exponents)
        self.feature_scale: Optional[Array] = None
        self.valid_mask: Optional[Array] = None
        self.structural_groups: Dict[Support, Array] = {}
        self._build_structural_groups()

    def _generate_exponents(self):
        exponents = []
        for degree in range(1, self.max_polynomial_degree + 1):
            for combo in combinations_with_replacement(range(self.n_nodes), degree):
                exp = [0] * self.n_nodes
                for j in combo:
                    exp[j] += 1
                exponents.append(tuple(exp))
        return tuple(exponents)

    def _build_structural_groups(self) -> None:
        valid_mask = np.zeros((self.n_features, self.n_nodes), dtype=bool)
        groups = {}

        for a, exponent in enumerate(self.exponents):
            variables = {j + 1 for j, p in enumerate(exponent) if p > 0}
            for i in range(self.n_nodes):
                support = tuple(sorted(variables | {i + 1}))
                if len(support) > self.max_interaction_order:
                    continue
                valid_mask[a, i] = True
                flat = a * self.n_nodes + i
                groups.setdefault(support, []).append(flat)

        self.valid_mask = valid_mask
        self.structural_groups = {
            s: np.asarray(idx, dtype=int)
            for s, idx in sorted(groups.items(), key=lambda kv: (len(kv[0]), kv[0]))
        }

    def evaluate_raw(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_nodes:
            raise ValueError(f"X must have shape (n_samples, {self.n_nodes}).")

        Theta = np.ones((X.shape[0], self.n_features), dtype=float)
        for a, exponent in enumerate(self.exponents):
            values = np.ones(X.shape[0], dtype=float)
            for j, power in enumerate(exponent):
                if power:
                    values *= X[:, j] ** power
            Theta[:, a] = values
        return Theta

    def fit_transform(self, X: Array) -> Array:
        raw = self.evaluate_raw(X)
        scale = np.sqrt(np.mean(raw ** 2, axis=0))
        scale = np.where(scale > 0.0, scale, 1.0)
        self.feature_scale = scale
        return raw / scale[None, :]

    def transform(self, X: Array) -> Array:
        if self.feature_scale is None:
            raise RuntimeError("Call fit_transform on training data first.")
        return self.evaluate_raw(X) / self.feature_scale[None, :]

    def to_raw_coefficients(self, B_scaled: Array) -> Array:
        if self.feature_scale is None:
            raise RuntimeError("Library has not been fitted.")
        B_raw = np.asarray(B_scaled, dtype=float) / self.feature_scale[:, None]
        B_raw = B_raw.copy()
        B_raw[~self.valid_mask] = 0.0
        return B_raw

    def derivative_design(self, X: Array, variable_index: int) -> Array:
        X = np.asarray(X, dtype=float)
        j = int(variable_index)
        D = np.zeros((X.shape[0], self.n_features), dtype=float)

        for a, exponent in enumerate(self.exponents):
            p_j = exponent[j]
            if p_j == 0:
                continue
            values = np.ones(X.shape[0], dtype=float)
            for k, power in enumerate(exponent):
                dp = power - (1 if k == j else 0)
                if dp:
                    values *= X[:, k] ** dp
            D[:, a] = p_j * values
        return D


def _coerce_forward_data(
    XF: Union[Array, Mapping[float, Array]],
    eps: Sequence[float],
) -> Array:
    eps = np.asarray(eps, dtype=float)

    if isinstance(XF, Mapping):
        keys = np.asarray(list(XF.keys()), dtype=float)
        rows = []
        for e in eps:
            if e in XF:
                arr = XF[e]
            else:
                idx = int(np.argmin(np.abs(keys - e)))
                if not np.isclose(keys[idx], e, rtol=1e-12, atol=1e-15):
                    raise KeyError(f"No forward-data entry for epsilon={e}.")
                arr = XF[list(XF.keys())[idx]]
            rows.append(np.asarray(arr, dtype=float))
        return np.stack(rows, axis=0)

    out = np.asarray(XF, dtype=float)
    if out.ndim != 3:
        raise ValueError("XF ndarray must have shape (n_eps, n_samples, n_nodes).")
    if out.shape[0] != len(eps):
        raise ValueError("XF first axis must align with eps.")
    return out


def _relative_field_error(prediction: Array, target: Array) -> float:
    denom = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
    return float(np.linalg.norm(prediction - target) / denom)


def _field_rms(X: Array) -> float:
    return float(np.sqrt(np.mean(np.asarray(X, dtype=float) ** 2)))


def _resolution_extrapolation(
    X0: Array,
    XF: Array,
    eps: Array,
    n_points: int = 4,
    polynomial_order: int = 3,
):
    X0 = np.asarray(X0, dtype=float)
    XF = np.asarray(XF, dtype=float)
    eps = np.asarray(eps, dtype=float)

    if len(eps) < n_points:
        raise ValueError(f"At least {n_points} epsilon values are required.")

    chosen = np.argsort(eps)[:n_points]
    eps_small = eps[chosen]
    XF_small = XF[chosen]

    if np.any(eps_small <= 0):
        raise ValueError("All epsilon values must be positive.")

    Y = (XF_small - X0[None, :, :]) / eps_small[:, None, None]
    eps_scale = float(np.max(eps_small))
    z = eps_small / eps_scale
    V = np.column_stack([z ** p for p in range(polynomial_order + 1)])

    Vinv = np.linalg.inv(V) if V.shape[0] == V.shape[1] else np.linalg.pinv(V)
    coeff = np.einsum("ab,bmn->amn", Vinv, Y)

    A = coeff[0]
    G1 = coeff[1] / eps_scale
    return A, G1, eps_small


class _AdaptiveGroupLasso:
    """
    Adaptive Group LASSO with a stable Ridge pilot and KKT-certified
    working-set optimization.

    The Ridge estimate is used only to construct adaptive group weights.
    Final coefficients are still obtained by an unpenalized post-selection
    least-squares refit, preserving the v2 interpretation of reported beta_S.
    """

    def __init__(
        self,
        library: StructuralPolynomialLibrary,
        n_lambdas: int,
        lambda_min_ratio: float,
        gamma: float = 1.0,
        delta_ratio: float = 1e-8,
        max_iter: int = 5000,
        tol: float = 1e-8,
        active_group_tol: float = 1e-10,
        ridge_condition_target: float = 1e6,
        ridge_alpha_floor_ratio: float = 1e-12,
        kkt_tol: float = 1e-7,
        max_kkt_expansions: int = 100,
        verbose: bool = True,
        label: str = "generator",
    ):
        self.library = library
        self.n_lambdas = int(n_lambdas)
        self.lambda_min_ratio = float(lambda_min_ratio)
        self.gamma = float(gamma)
        self.delta_ratio = float(delta_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.active_group_tol = float(active_group_tol)
        self.ridge_condition_target = float(ridge_condition_target)
        self.ridge_alpha_floor_ratio = float(ridge_alpha_floor_ratio)
        self.kkt_tol = float(kkt_tol)
        self.max_kkt_expansions = int(max_kkt_expansions)
        self.verbose = bool(verbose)
        self.label = str(label)
        self.groups = library.structural_groups
        self.valid_mask = library.valid_mask

        if self.ridge_condition_target <= 1.0:
            raise ValueError("ridge_condition_target must be > 1.")
        if self.ridge_alpha_floor_ratio <= 0.0:
            raise ValueError("ridge_alpha_floor_ratio must be positive.")
        if self.kkt_tol < 0.0:
            raise ValueError("kkt_tol must be non-negative.")
        if self.max_kkt_expansions < 1:
            raise ValueError("max_kkt_expansions must be >= 1.")

    @staticmethod
    def _power_largest_eigenvalue_design(
        X: Array,
        *,
        max_iter: int = 80,
        tol: float = 1e-8,
    ) -> float:
        """Estimate lambda_max(X^T X / n) without explicitly forming X^T X."""
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        if n == 0 or p == 0:
            return 0.0

        # Work in the smaller primal/dual space.
        if p <= n:
            v = np.ones(p, dtype=float)
            v /= np.linalg.norm(v)
            last = 0.0
            for _ in range(max_iter):
                z = X.T @ (X @ v) / n
                norm = float(np.linalg.norm(z))
                if norm == 0.0:
                    return 0.0
                v = z / norm
                rayleigh = float(v @ (X.T @ (X @ v)) / n)
                if abs(rayleigh - last) <= tol * max(abs(rayleigh), 1.0):
                    return max(rayleigh, 0.0)
                last = rayleigh
            return max(last, 0.0)

        v = np.ones(n, dtype=float)
        v /= np.linalg.norm(v)
        last = 0.0
        for _ in range(max_iter):
            z = X @ (X.T @ v) / n
            norm = float(np.linalg.norm(z))
            if norm == 0.0:
                return 0.0
            v = z / norm
            rayleigh = float(v @ (X @ (X.T @ v)) / n)
            if abs(rayleigh - last) <= tol * max(abs(rayleigh), 1.0):
                return max(rayleigh, 0.0)
            last = rayleigh
        return max(last, 0.0)

    def _adaptive_ridge_alpha(self, X: Array) -> Tuple[float, float, float]:
        """
        Choose the smallest scale-aware Ridge alpha that bounds the condition
        number of X^T X / n + alpha I by ridge_condition_target.

        Returns
        -------
        alpha, lambda_max, lambda_min
        """
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        if p == 0:
            return 0.0, 0.0, 0.0

        lambda_max = self._power_largest_eigenvalue_design(X)
        if lambda_max <= 0.0:
            return self.ridge_alpha_floor_ratio, 0.0, 0.0

        if p >= n:
            # X^T X is necessarily singular in feature space.
            lambda_min = 0.0
        else:
            # p < n in this branch; the p x p Gram matrix is the smaller one.
            gram = (X.T @ X) / n
            eigvals = np.linalg.eigvalsh(gram)
            lambda_min = max(float(eigvals[0]), 0.0)
            lambda_max = max(float(eigvals[-1]), lambda_max)

        kappa = self.ridge_condition_target
        alpha_cond = max(
            0.0,
            (lambda_max - kappa * lambda_min) / (kappa - 1.0),
        )
        alpha_floor = self.ridge_alpha_floor_ratio * max(lambda_max, 1.0)
        alpha = max(alpha_cond, alpha_floor)
        return float(alpha), float(lambda_max), float(lambda_min)

    def _ridge_solve(self, X: Array, Y: Array, alpha: float) -> Array:
        """Solve the normalized Ridge problem using primal or dual algebra."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]

        n, p = X.shape
        if p <= n:
            lhs = X.T @ X
            lhs.flat[:: p + 1] += n * alpha
            rhs = X.T @ Y
            return np.linalg.solve(lhs, rhs)

        # Dual form avoids a p x p system when p > n.
        lhs = X @ X.T
        lhs.flat[:: n + 1] += n * alpha
        dual = np.linalg.solve(lhs, Y)
        return X.T @ dual

    def _initial_ridge(self, Theta: Array, Y: Array):
        """Condition-number-adaptive Ridge pilot for adaptive group weights."""
        n_features = Theta.shape[1]
        n_outputs = Y.shape[1]
        B = np.zeros((n_features, n_outputs), dtype=float)
        alphas = np.full(n_outputs, np.nan, dtype=float)

        # Fast path: every output uses the same full design matrix.
        if bool(np.all(self.valid_mask)):
            alpha, _, _ = self._adaptive_ridge_alpha(Theta)
            B[:, :] = self._ridge_solve(Theta, Y, alpha)
            alphas[:] = alpha
            return B, alphas

        # General path for support-restricted libraries/masks.
        cache = {}
        for i in range(n_outputs):
            features = np.flatnonzero(self.valid_mask[:, i])
            if len(features) == 0:
                continue

            key = features.tobytes()
            if key not in cache:
                X_i = Theta[:, features]
                alpha, _, _ = self._adaptive_ridge_alpha(X_i)
                cache[key] = (features, X_i, alpha, [])
            cache[key][3].append(i)

        for features, X_i, alpha, outputs in cache.values():
            beta = self._ridge_solve(X_i, Y[:, outputs], alpha)
            B[np.ix_(features, outputs)] = beta
            alphas[outputs] = alpha

        return B, alphas

    def _adaptive_weights(self, B_init: Array):
        flat = B_init.ravel(order="C")
        norms = {
            s: float(np.linalg.norm(flat[idx]))
            for s, idx in self.groups.items()
        }
        max_norm = max(norms.values()) if norms else 0.0
        delta = max(1e-12, self.delta_ratio * max_norm)
        return {
            s: np.sqrt(len(idx)) / (norms[s] + delta) ** self.gamma
            for s, idx in self.groups.items()
        }

    def _full_gradient(self, Theta: Array, residual: Array) -> Array:
        """Gradient of 1/(2n)||Theta B - Y||_F^2 on all valid directions."""
        n = Theta.shape[0]
        if bool(np.all(self.valid_mask)):
            return (Theta.T @ residual) / n

        gradient = np.zeros((Theta.shape[1], residual.shape[1]), dtype=float)
        for i in range(residual.shape[1]):
            features = np.flatnonzero(self.valid_mask[:, i])
            if len(features):
                gradient[features, i] = (Theta[:, features].T @ residual[:, i]) / n
        return gradient

    def _lambda_max(self, Theta: Array, Y: Array, weights) -> float:
        grad0 = self._full_gradient(Theta, -Y)
        flat = grad0.ravel(order="C")
        return float(max(
            np.linalg.norm(flat[idx]) / weights[s]
            for s, idx in self.groups.items()
        ))

    def _working_indices_by_output(self, supports):
        n_outputs = self.valid_mask.shape[1]
        if not supports:
            return [np.empty(0, dtype=int) for _ in range(n_outputs)]

        active_flat = np.concatenate([self.groups[s] for s in supports])
        out = []
        for i in range(n_outputs):
            flat_i = active_flat[active_flat % n_outputs == i]
            out.append(np.unique(flat_i // n_outputs))
        return out

    def _predict_restricted(self, Theta: Array, B: Array, features_by_output) -> Array:
        pred = np.zeros((Theta.shape[0], B.shape[1]), dtype=float)
        for i, features in enumerate(features_by_output):
            if len(features):
                pred[:, i] = Theta[:, features] @ B[features, i]
        return pred

    def _prox_working_set(self, B: Array, scale: float, weights, working_set) -> Array:
        out = np.zeros_like(B)
        if not working_set:
            return out

        flat_in = B.ravel(order="C")
        flat_out = out.ravel(order="C")
        for s in working_set:
            idx = self.groups[s]
            vec = flat_in[idx]
            norm = float(np.linalg.norm(vec))
            threshold = scale * weights[s]
            if norm > threshold:
                flat_out[idx] = vec * (1.0 - threshold / norm)
        return flat_out.reshape(out.shape, order="C")

    def _solve_restricted(
        self,
        Theta: Array,
        Y: Array,
        lam: float,
        weights,
        B_start: Array,
        step_size: float,
        working_set,
    ):
        working_set = tuple(sorted(set(working_set), key=lambda s: (len(s), s)))
        features_by_output = self._working_indices_by_output(working_set)

        B = np.zeros_like(B_start)
        if working_set:
            flat_B = B.ravel(order="C")
            flat_start = B_start.ravel(order="C")
            active_flat = np.concatenate([self.groups[s] for s in working_set])
            flat_B[active_flat] = flat_start[active_flat]
            B = flat_B.reshape(B.shape, order="C")

        converged = False
        relative_change = np.inf

        if not working_set:
            return B, {
                "iterations": 0,
                "converged": True,
                "relative_change": 0.0,
            }

        active_flat = np.concatenate([self.groups[s] for s in working_set])

        for iteration in range(1, self.max_iter + 1):
            pred = self._predict_restricted(Theta, B, features_by_output)
            residual = pred - Y

            proposal = B.copy()
            for i, features in enumerate(features_by_output):
                if len(features):
                    grad_i = (Theta[:, features].T @ residual[:, i]) / Theta.shape[0]
                    proposal[features, i] -= step_size * grad_i

            B_new = self._prox_working_set(
                proposal,
                step_size * lam,
                weights,
                working_set,
            )

            flat_old = B.ravel(order="C")[active_flat]
            flat_new = B_new.ravel(order="C")[active_flat]
            denom = max(float(np.linalg.norm(flat_old)), 1e-12)
            relative_change = float(np.linalg.norm(flat_new - flat_old) / denom)
            B = B_new
            if relative_change < self.tol:
                converged = True
                break

        return B, {
            "iterations": iteration,
            "converged": converged,
            "relative_change": relative_change,
        }

    def _kkt_violations(self, Theta, Y, B, lam, weights, working_set):
        """
        Check omitted groups against the Group-LASSO KKT condition.

        An omitted group S must satisfy ||grad_S||_2 <= lambda * w_S.
        Violating groups are returned for reactivation.
        """
        features_by_output = self._working_indices_by_output(working_set)
        residual = self._predict_restricted(Theta, B, features_by_output) - Y
        gradient = self._full_gradient(Theta, residual)
        flat = gradient.ravel(order="C")

        working = set(working_set)
        violations = []
        max_ratio = 0.0
        for s, idx in self.groups.items():
            if s in working:
                continue
            threshold = lam * weights[s]
            norm = float(np.linalg.norm(flat[idx]))
            if threshold <= 0.0:
                ratio = np.inf if norm > 0.0 else 0.0
            else:
                ratio = norm / threshold
            max_ratio = max(max_ratio, ratio)
            if ratio > 1.0 + self.kkt_tol:
                violations.append((s, ratio))

        violations.sort(key=lambda item: item[1], reverse=True)
        return violations, float(max_ratio)

    def _solve_one_lambda(self, Theta, Y, lam, weights, B_start, step_size, working_set):
        working = set(working_set)
        B = B_start.copy()
        total_iterations = 0
        all_converged = True
        final_relative_change = np.inf
        total_reactivated = 0
        expansions = 0
        final_max_kkt_ratio = np.inf

        while True:
            B, info = self._solve_restricted(
                Theta,
                Y,
                lam,
                weights,
                B,
                step_size,
                working,
            )
            total_iterations += int(info["iterations"])
            all_converged = all_converged and bool(info["converged"])
            final_relative_change = float(info["relative_change"])

            violations, max_ratio = self._kkt_violations(
                Theta,
                Y,
                B,
                lam,
                weights,
                working,
            )
            final_max_kkt_ratio = max_ratio
            if not violations:
                break

            expansions += 1
            if expansions > self.max_kkt_expansions:
                raise RuntimeError(
                    f"{self.label}: exceeded max_kkt_expansions="
                    f"{self.max_kkt_expansions} at lambda={lam:.3e}."
                )

            new_supports = [s for s, _ in violations]
            total_reactivated += len(new_supports)
            working.update(new_supports)

        return B, tuple(sorted(working, key=lambda s: (len(s), s))), {
            "iterations": total_iterations,
            "converged": all_converged,
            "relative_change": final_relative_change,
            "kkt_expansions": expansions,
            "kkt_reactivations": total_reactivated,
            "working_set_size": len(working),
            "max_kkt_ratio": final_max_kkt_ratio,
            "kkt_certified": (
                all_converged
                and final_max_kkt_ratio <= 1.0 + self.kkt_tol
            ),
        }

    def _active_supports(self, B: Array, candidate_supports=None):
        flat = B.ravel(order="C")
        if candidate_supports is None:
            items = self.groups.items()
        else:
            items = ((s, self.groups[s]) for s in candidate_supports)
        active = [
            s for s, idx in items
            if np.linalg.norm(flat[idx]) > self.active_group_tol
        ]
        return tuple(sorted(active, key=lambda s: (len(s), s)))

    def _refit(self, Theta: Array, Y: Array, supports):
        """Unpenalized post-selection refit; intentionally retained from v2."""
        B = np.zeros((Theta.shape[1], Y.shape[1]), dtype=float)
        if not supports:
            return B
        active_flat = np.concatenate([self.groups[s] for s in supports])

        for i in range(Y.shape[1]):
            flat_i = active_flat[active_flat % Y.shape[1] == i]
            if len(flat_i) == 0:
                continue
            features = np.unique(flat_i // Y.shape[1])
            beta, *_ = np.linalg.lstsq(Theta[:, features], Y[:, i], rcond=None)
            B[features, i] = beta

        B[~self.valid_mask] = 0.0
        return B

    def _stats(self, Theta, Y, B, supports):
        pred = Theta @ B
        residual = Y - pred
        rss = float(np.sum(residual ** 2))
        n_parameters = int(sum(len(self.groups[s]) for s in supports))
        n_obs = int(Y.shape[0] * Y.shape[1])
        rss_safe = max(rss, np.finfo(float).tiny)
        bic = n_obs * np.log(rss_safe / n_obs) + n_parameters * np.log(n_obs)
        return {
            "rss": rss,
            "bic": float(bic),
            "n_parameters": n_parameters,
            "relative_error": _relative_field_error(pred, Y),
        }

    def fit(self, Theta_train, Y_train, Theta_test=None, Y_test=None):
        B_init, ridge_alphas = self._initial_ridge(Theta_train, Y_train)
        weights = self._adaptive_weights(B_init)
        lambda_max = self._lambda_max(Theta_train, Y_train, weights)
        if lambda_max <= 0:
            raise RuntimeError(f"{self.label}: lambda_max is non-positive.")

        lambda_path = lambda_max * np.geomspace(
            1.0, self.lambda_min_ratio, self.n_lambdas
        )

        # Conservative global step size. Power iteration avoids a full SVD.
        lipschitz = self._power_largest_eigenvalue_design(Theta_train)
        if lipschitz <= 0.0:
            raise RuntimeError(f"{self.label}: non-positive Lipschitz constant.")
        step_size = 1.0 / lipschitz

        B_warm = np.zeros_like(B_init)
        working_set = tuple()
        cache = {}
        rows = []
        start = time.perf_counter()
        progress_every = max(1, self.n_lambdas // 16)
        total_reactivations = 0

        finite_alphas = ridge_alphas[np.isfinite(ridge_alphas)]
        alpha_min = float(np.min(finite_alphas)) if len(finite_alphas) else np.nan
        alpha_max = float(np.max(finite_alphas)) if len(finite_alphas) else np.nan

        if self.verbose:
            print(
                f"Starting KKT-certified working-set Adaptive Group LASSO "
                f"for {self.label} ({self.n_lambdas} lambdas)..."
            )
            print(
                f"  pilot = adaptive Ridge | alpha range = "
                f"[{alpha_min:.3e}, {alpha_max:.3e}]"
            )

        for path_index, lam in enumerate(lambda_path):
            t0 = time.perf_counter()
            B_pen, working_set, info = self._solve_one_lambda(
                Theta_train,
                Y_train,
                float(lam),
                weights,
                B_warm,
                step_size,
                working_set,
            )
            B_warm = B_pen
            total_reactivations += int(info["kkt_reactivations"])
            supports = self._active_supports(B_pen, working_set)

            if supports not in cache:
                B_refit = self._refit(Theta_train, Y_train, supports)
                stats = self._stats(Theta_train, Y_train, B_refit, supports)
                cache[supports] = {"B_refit": B_refit, "stats": stats}
            else:
                stats = cache[supports]["stats"]

            rows.append({
                "path_index": path_index,
                "lambda": float(lam),
                "n_groups": len(supports),
                "n_parameters": stats["n_parameters"],
                "train_relative_error": stats["relative_error"],
                "rss": stats["rss"],
                "bic": stats["bic"],
                "solver_iterations": info["iterations"],
                "solver_converged": info["converged"],
                "solver_relative_change": info["relative_change"],
                "working_set_size": info["working_set_size"],
                "kkt_expansions": info["kkt_expansions"],
                "kkt_reactivations": info["kkt_reactivations"],
                "max_kkt_ratio": info["max_kkt_ratio"],
                "kkt_certified": info["kkt_certified"],
                "lambda_runtime": float(time.perf_counter() - t0),
                "active_supports": supports,
            })

            if self.verbose and (
                path_index % progress_every == 0
                or path_index == len(lambda_path) - 1
            ):
                print(
                    f"  [{path_index + 1:03d}/{len(lambda_path)}] "
                    f"lambda={lam:.3e} | groups={len(supports):3d} | "
                    f"WS={info['working_set_size']:3d} | "
                    f"KKT+={info['kkt_reactivations']:3d} | "
                    f"iter={info['iterations']:4d} | "
                    f"elapsed={time.perf_counter() - start:.1f}s"
                )

        ledger = pd.DataFrame(rows)
        best_idx = ledger["bic"].idxmin()
        best = ledger.loc[best_idx]
        supports = best["active_supports"]
        B_scaled = cache[supports]["B_refit"]
        B_raw = self.library.to_raw_coefficients(B_scaled)

        train_error = _relative_field_error(Theta_train @ B_scaled, Y_train)
        test_error = None
        if Theta_test is not None and Y_test is not None:
            test_error = _relative_field_error(Theta_test @ B_scaled, Y_test)

        counts = dict(sorted(Counter(len(s) for s in supports).items()))
        all_kkt = bool(ledger["kkt_certified"].all())
        max_final_ratio = float(ledger["max_kkt_ratio"].max())

        return GeneratorFitResult(
            coefficients_scaled=B_scaled,
            coefficients_raw=B_raw,
            selected_supports=supports,
            train_relative_error=train_error,
            test_relative_error=test_error,
            lambda_selected=float(best["lambda"]),
            path_ledger=ledger,
            group_counts_by_size=counts,
            all_solver_runs_converged=bool(ledger["solver_converged"].all()),
            max_solver_iterations=int(ledger["solver_iterations"].max()),
            pilot_method="condition-number-adaptive Ridge",
            ridge_alpha_min=alpha_min,
            ridge_alpha_max=alpha_max,
            kkt_certified=all_kkt,
            total_kkt_reactivations=total_reactivations,
            max_final_kkt_ratio=max_final_ratio,
        )


def _finite_flow_correction(X, library, B_raw):
    Theta_raw = library.evaluate_raw(X)
    F = Theta_raw @ B_raw
    Q = np.zeros_like(F)

    for j in range(library.n_nodes):
        Dtheta = library.derivative_design(X, j)
        J_column = Dtheta @ B_raw
        Q += 0.5 * J_column * F[:, j][:, None]

    return Q


class TSCInference:
    """
    Forward-only TSC inference with three public modelling choices.

    The remaining estimator settings are internal/frozen. v3 keeps the
    public modelling interface unchanged while using a condition-adaptive
    Ridge pilot and KKT-certified working-set AGLASSO internally.
    """

    _N_EXTRAPOLATION_POINTS = 4
    _EXTRAPOLATION_POLY_ORDER = 3

    _ADAPTIVE_GAMMA = 1.0
    _ADAPTIVE_DELTA_RATIO = 1e-8
    _MAX_ITER = 5000
    _SOLVER_TOL = 1e-8
    _ACTIVE_GROUP_TOL = 1e-10

    _RIDGE_CONDITION_TARGET = 1e6
    _RIDGE_ALPHA_FLOOR_RATIO = 1e-12
    _KKT_TOL = 1e-7
    _MAX_KKT_EXPANSIONS = 100

    _F0_N_LAMBDAS = 80
    _F0_LAMBDA_MIN_RATIO = 1e-6

    _F1_N_LAMBDAS = 120
    _F1_LAMBDA_MIN_RATIO = 1e-10

    def __init__(
        self,
        max_interaction_order: int = 4,
        max_polynomial_degree: int = 3,
        temporal_order: int = 1,
        verbose: bool = True,
    ):
        self.max_interaction_order = int(max_interaction_order)
        self.max_polynomial_degree = int(max_polynomial_degree)
        self.temporal_order = int(temporal_order)
        self.verbose = bool(verbose)

        if self.temporal_order != 1:
            raise NotImplementedError(
                "TSCInference v3 currently supports temporal_order=1 only "
                "(recovery of F^(0) and F^(1))."
            )

        self.library_: Optional[StructuralPolynomialLibrary] = None
        self.result_: Optional[TSCResult] = None

    def fit(
        self,
        X0: Array,
        XF: Union[Array, Mapping[float, Array]],
        eps: Sequence[float],
        *,
        X0_test: Optional[Array] = None,
        XF_test: Optional[Union[Array, Mapping[float, Array]]] = None,
    ) -> TSCResult:
        X0 = np.asarray(X0, dtype=float)
        eps = np.asarray(eps, dtype=float)

        if X0.ndim != 2:
            raise ValueError("X0 must have shape (n_samples, n_nodes).")
        if eps.ndim != 1:
            raise ValueError("eps must be one-dimensional.")
        if len(eps) < self._N_EXTRAPOLATION_POINTS:
            raise ValueError(
                f"At least {self._N_EXTRAPOLATION_POINTS} epsilon values are required."
            )

        XF_array = _coerce_forward_data(XF, eps)
        if XF_array.shape[1:] != X0.shape:
            raise ValueError(
                "XF must have shape (n_eps, n_samples, n_nodes) matching X0."
            )

        have_test = X0_test is not None or XF_test is not None
        if have_test:
            if X0_test is None or XF_test is None:
                raise ValueError("Provide both X0_test and XF_test, or neither.")
            X0_test = np.asarray(X0_test, dtype=float)
            XF_test_array = _coerce_forward_data(XF_test, eps)
            if X0_test.ndim != 2 or X0_test.shape[1] != X0.shape[1]:
                raise ValueError("X0_test has incompatible shape.")
            if XF_test_array.shape[1:] != X0_test.shape:
                raise ValueError("XF_test has incompatible shape.")
        else:
            XF_test_array = None

        # 1. Multi-resolution extrapolation
        A_train, G1_train, eps_used = _resolution_extrapolation(
            X0,
            XF_array,
            eps,
            n_points=self._N_EXTRAPOLATION_POINTS,
            polynomial_order=self._EXTRAPOLATION_POLY_ORDER,
        )

        if have_test:
            A_test, G1_test, eps_used_test = _resolution_extrapolation(
                X0_test,
                XF_test_array,
                eps,
                n_points=self._N_EXTRAPOLATION_POINTS,
                polynomial_order=self._EXTRAPOLATION_POLY_ORDER,
            )
            if not np.allclose(eps_used, eps_used_test):
                raise RuntimeError("Train/test extrapolation grids differ.")
        else:
            A_test = None
            G1_test = None

        # 2. Generic structural library
        library = StructuralPolynomialLibrary(
            n_nodes=X0.shape[1],
            max_polynomial_degree=self.max_polynomial_degree,
            max_interaction_order=self.max_interaction_order,
        )
        self.library_ = library
        Theta_train = library.fit_transform(X0)
        Theta_test = library.transform(X0_test) if have_test else None

        if self.verbose:
            print("TSC configuration:")
            print(f"  max_interaction_order = {self.max_interaction_order}")
            print(f"  max_polynomial_degree = {self.max_polynomial_degree}")
            print(f"  temporal_order = {self.temporal_order}")
            print(f"  extrapolation eps = {np.array2string(eps_used, precision=6)}")
            print(f"  library features/output = {library.n_features}")
            print(f"  structural groups = {len(library.structural_groups)}")
            print()

        # 3. F^(0)
        estimator_F0 = _AdaptiveGroupLasso(
            library=library,
            n_lambdas=self._F0_N_LAMBDAS,
            lambda_min_ratio=self._F0_LAMBDA_MIN_RATIO,
            gamma=self._ADAPTIVE_GAMMA,
            delta_ratio=self._ADAPTIVE_DELTA_RATIO,
            max_iter=self._MAX_ITER,
            tol=self._SOLVER_TOL,
            active_group_tol=self._ACTIVE_GROUP_TOL,
            ridge_condition_target=self._RIDGE_CONDITION_TARGET,
            ridge_alpha_floor_ratio=self._RIDGE_ALPHA_FLOOR_RATIO,
            kkt_tol=self._KKT_TOL,
            max_kkt_expansions=self._MAX_KKT_EXPANSIONS,
            verbose=self.verbose,
            label="F^(0)",
        )
        F0 = estimator_F0.fit(Theta_train, A_train, Theta_test, A_test)

        # 4. Finite-flow correction
        Q_train = _finite_flow_correction(X0, library, F0.coefficients_raw)
        F1_target_train = G1_train - Q_train

        if have_test:
            Q_test = _finite_flow_correction(X0_test, library, F0.coefficients_raw)
            F1_target_test = G1_test - Q_test
        else:
            Q_test = None
            F1_target_test = None

        # 5. F^(1)
        estimator_F1 = _AdaptiveGroupLasso(
            library=library,
            n_lambdas=self._F1_N_LAMBDAS,
            lambda_min_ratio=self._F1_LAMBDA_MIN_RATIO,
            gamma=self._ADAPTIVE_GAMMA,
            delta_ratio=self._ADAPTIVE_DELTA_RATIO,
            max_iter=self._MAX_ITER,
            tol=self._SOLVER_TOL,
            active_group_tol=self._ACTIVE_GROUP_TOL,
            ridge_condition_target=self._RIDGE_CONDITION_TARGET,
            ridge_alpha_floor_ratio=self._RIDGE_ALPHA_FLOOR_RATIO,
            kkt_tol=self._KKT_TOL,
            max_kkt_expansions=self._MAX_KKT_EXPANSIONS,
            verbose=self.verbose,
            label="F^(1)",
        )
        F1 = estimator_F1.fit(
            Theta_train,
            F1_target_train,
            Theta_test,
            F1_target_test,
        )

        diagnostics = {
            "RMS(G1_data)": _field_rms(G1_train),
            "RMS(Q[F0_hat])": _field_rms(Q_train),
            "RMS(F1_target)": _field_rms(F1_target_train),
            "RMS(Q)/RMS(G1)": (
                _field_rms(Q_train)
                / max(_field_rms(G1_train), np.finfo(float).tiny)
            ),
        }

        result = TSCResult(
            F0=F0,
            F1=F1,
            extrapolation_eps=eps_used,
            A_train=A_train,
            G1_train=G1_train,
            Q_train=Q_train,
            F1_target_train=F1_target_train,
            A_test=A_test,
            G1_test=G1_test,
            Q_test=Q_test,
            F1_target_test=F1_target_test,
            diagnostics=diagnostics,
            _library=library,
        )
        self.result_ = result

        if self.verbose:
            print()
            result.summary(show_supports=False)

        return result


__all__ = ["TSCInference", "TSCResult", "GeneratorFitResult"]
