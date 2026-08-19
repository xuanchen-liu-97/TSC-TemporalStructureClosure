"""
TSC_AGLASSO.py
==============

Forward-only Temporal Structural Closure (TSC) inference.

Internal implementation: v3.6
- condition-number-adaptive Ridge pilot for Adaptive Group LASSO weights
- KKT-certified working-set optimization with support reactivation
- KKT-event-driven adaptive regularization path learned from omitted-group entry scores
- data-derived extrapolation-uncertainty KKT stopping for structural certification
- high-recall AGLASSO screening with contribution pruning retained as fallback
- validation-aware early stopping before the interpolation regime
- saturated post-selection OLS guard
- corrected final-vs-intermediate KKT convergence bookkeeping
- completed-F^(0) checkpoint / resume support
- lambda-boundary partial-F^(1) checkpoint / resume support

v3.6 keeps the Adaptive-Ridge + KKT working-set Group-LASSO core and the
KKT-event-driven continuation from v3.5, and adds a data-derived stopping
certificate for F^(1). The central temporal target is still the frozen
4-point cubic extrapolation. Two higher-order local extrapolants (5-point
quartic and 6-point quintic) are used only to estimate the finite-resolution
uncertainty of that target. Conditional on each current support S, this
uncertainty field is projected onto S and converted to the same omitted-group
KKT entry scale as the data residual. The path stops when the strongest
unresolved structural event is no larger than the uncertainty-induced KKT
entry envelope. Thus lambda continuation remains data driven, while the
stopping point is learned from the resolution of the endpoint data rather
than from an oracle or benchmark-specific lambda_min. Contribution pruning is
retained only as a fallback when no uncertainty certificate is available.
Training BIC remains diagnostic only.

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
from pathlib import Path
import gzip
import hashlib
import os
import pickle
import time

import numpy as np
import pandas as pd

Array = np.ndarray
Support = Tuple[int, ...]

_IMPLEMENTATION_VERSION = "v3.6"
_F1_PARTIAL_CHECKPOINT_FORMAT_VERSION = 5


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
    all_path_kkt_certified: bool = False
    validation_relative_error: float = np.nan
    validation_mse: float = np.nan
    selection_method: str = "validation-1se"
    n_fit_samples: int = 0
    n_validation_samples: int = 0
    early_stop_reason: str = ""
    path_boundary_hit: bool = False
    total_kkt_reactivations: int = 0
    max_final_kkt_ratio: float = np.nan
    screening_supports: Tuple[Support, ...] = ()
    screening_lambda: float = np.nan
    screening_validation_relative_error: float = np.nan
    screening_group_counts_by_size: Optional[Dict[int, int]] = None
    pruning_ledger: Optional[pd.DataFrame] = None
    pruning_applied: bool = False
    pruning_threshold: float = np.nan
    pruning_score_method: str = "none"
    final_restricted_converged: bool = False
    final_kkt_satisfied: bool = False
    all_intermediate_solver_runs_converged: bool = False
    uncertainty_certified: bool = False
    unresolved_kkt_score: float = np.nan
    uncertainty_kkt_floor: float = np.nan
    uncertainty_kkt_ratio: float = np.nan
    uncertainty_kkt_levels: Tuple[float, ...] = ()
    uncertainty_next_support: Optional[Support] = None

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
    F1_uncertainty_fields_train: Optional[Tuple[Array, ...]] = None
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
            f"  screening-lambda KKT certified = {self.F0.kkt_certified} "
            f"| final restricted converged = {self.F0.final_restricted_converged} "
            f"| final global KKT = {self.F0.final_kkt_satisfied}"
        )
        print(
            f"  all-path KKT = {self.F0.all_path_kkt_certified} "
            f"| reactivations = {self.F0.total_kkt_reactivations}"
        )
        print(
            f"  internal validation relative error = "
            f"{self.F0.validation_relative_error:.3e}"
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
        if self.F1.screening_supports:
            print(
                f"  AGLASSO screening groups = {len(self.F1.screening_supports)} "
                f"| screening by support size = "
                f"{self.F1.screening_group_counts_by_size}"
            )
            if self.F1.pruning_applied:
                print(
                    f"  pruning retained = {len(self.F1.selected_supports)}/"
                    f"{len(self.F1.screening_supports)} groups "
                    f"| effect threshold = {self.F1.pruning_threshold:.3e}"
                )
        print(
            f"  screening-lambda KKT certified = {self.F1.kkt_certified} "
            f"| final restricted converged = {self.F1.final_restricted_converged} "
            f"| final global KKT = {self.F1.final_kkt_satisfied}"
        )
        print(
            f"  all-path KKT = {self.F1.all_path_kkt_certified} "
            f"| reactivations = {self.F1.total_kkt_reactivations}"
        )
        if self.F1.uncertainty_certified:
            print(
                "  data-derived uncertainty certificate = True "
                f"| unresolved KKT = {self.F1.unresolved_kkt_score:.3e} "
                f"| uncertainty floor = {self.F1.uncertainty_kkt_floor:.3e} "
                f"| ratio = {self.F1.uncertainty_kkt_ratio:.3f}"
            )
        print(
            f"  internal validation relative error = "
            f"{self.F1.validation_relative_error:.3e}"
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
        coarse_n_lambdas: Optional[int] = None,
        refine_n_lambdas: int = 0,
        gamma: float = 1.0,
        delta_ratio: float = 1e-8,
        max_iter: int = 5000,
        tol: float = 1e-8,
        active_group_tol: float = 1e-10,
        ridge_condition_target: float = 1e6,
        ridge_alpha_floor_ratio: float = 1e-12,
        kkt_tol: float = 1e-7,
        max_kkt_expansions: int = 100,
        validation_fraction: float = 0.20,
        validation_seed: int = 20260818,
        early_stop_min_evals: int = 8,
        early_stop_patience: int = 2,
        early_stop_relative_degradation: float = 0.02,
        early_stop_support_growth_ratio: float = 1.10,
        exact_fit_plateau_patience: int = 3,
        exact_fit_relative_error: float = 1e-8,
        # v3.6 KKT-event-driven lambda continuation. `lambda_min_ratio` is
        # retained only as an emergency numerical safety floor; it no longer
        # defines a predeclared logarithmic search path.
        adaptive_geometric_ratio: float = 0.75,
        adaptive_entry_fraction: float = 0.98,
        adaptive_max_jump_decades: float = 2.0,
        adaptive_max_evals: int = 60,
        enable_pruning: bool = False,
        pruning_n_candidates: int = 40,
        pruning_min_keep_fraction: float = 0.05,
        pruning_relative_tolerance: float = 0.02,
        verbose: bool = True,
        label: str = "generator",
    ):
        self.library = library
        self.n_lambdas = int(n_lambdas)
        self.lambda_min_ratio = float(lambda_min_ratio)
        self.coarse_n_lambdas = (
            self.n_lambdas
            if coarse_n_lambdas is None
            else int(coarse_n_lambdas)
        )
        self.refine_n_lambdas = int(refine_n_lambdas)
        self.gamma = float(gamma)
        self.delta_ratio = float(delta_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.active_group_tol = float(active_group_tol)
        self.ridge_condition_target = float(ridge_condition_target)
        self.ridge_alpha_floor_ratio = float(ridge_alpha_floor_ratio)
        self.kkt_tol = float(kkt_tol)
        self.max_kkt_expansions = int(max_kkt_expansions)
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        self.early_stop_min_evals = int(early_stop_min_evals)
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_relative_degradation = float(early_stop_relative_degradation)
        self.early_stop_support_growth_ratio = float(early_stop_support_growth_ratio)
        self.exact_fit_plateau_patience = int(exact_fit_plateau_patience)
        self.exact_fit_relative_error = float(exact_fit_relative_error)
        self.adaptive_geometric_ratio = float(adaptive_geometric_ratio)
        self.adaptive_entry_fraction = float(adaptive_entry_fraction)
        self.adaptive_max_jump_decades = float(adaptive_max_jump_decades)
        self.adaptive_max_evals = int(adaptive_max_evals)
        self.enable_pruning = bool(enable_pruning)
        self.pruning_n_candidates = int(pruning_n_candidates)
        self.pruning_min_keep_fraction = float(pruning_min_keep_fraction)
        self.pruning_relative_tolerance = float(pruning_relative_tolerance)
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
        if self.coarse_n_lambdas < 2:
            raise ValueError("coarse_n_lambdas must be >= 2.")
        if self.refine_n_lambdas < 0:
            raise ValueError("refine_n_lambdas must be >= 0.")
        if not (0.0 < self.lambda_min_ratio <= 1.0):
            raise ValueError("lambda_min_ratio must lie in (0, 1].")
        if not (0.0 < self.validation_fraction < 0.5):
            raise ValueError("validation_fraction must lie in (0, 0.5).")
        if self.early_stop_min_evals < 2:
            raise ValueError("early_stop_min_evals must be >= 2.")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be >= 1.")
        if self.early_stop_relative_degradation < 0.0:
            raise ValueError("early_stop_relative_degradation must be non-negative.")
        if self.early_stop_support_growth_ratio < 1.0:
            raise ValueError("early_stop_support_growth_ratio must be >= 1.")
        if self.exact_fit_plateau_patience < 2:
            raise ValueError("exact_fit_plateau_patience must be >= 2.")
        if self.exact_fit_relative_error <= 0.0:
            raise ValueError("exact_fit_relative_error must be positive.")
        if not (0.0 < self.adaptive_geometric_ratio < 1.0):
            raise ValueError("adaptive_geometric_ratio must lie in (0, 1).")
        if not (0.0 < self.adaptive_entry_fraction < 1.0):
            raise ValueError("adaptive_entry_fraction must lie in (0, 1).")
        if self.adaptive_max_jump_decades <= 0.0:
            raise ValueError("adaptive_max_jump_decades must be positive.")
        if self.adaptive_max_evals < 2:
            raise ValueError("adaptive_max_evals must be >= 2.")
        if self.pruning_n_candidates < 2:
            raise ValueError("pruning_n_candidates must be >= 2.")
        if not (0.0 < self.pruning_min_keep_fraction <= 1.0):
            raise ValueError("pruning_min_keep_fraction must lie in (0, 1].")
        if self.pruning_relative_tolerance < 0.0:
            raise ValueError("pruning_relative_tolerance must be non-negative.")

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

    def _next_inactive_entry_score(
        self,
        Theta: Array,
        Y: Array,
        B: Array,
        weights,
        working_set,
    ):
        """
        Estimate the next omitted-group KKT entry event.

        For an omitted group g, the Group-LASSO KKT condition is

            ||grad_g||_2 <= lambda * w_g.

        Holding the current residual fixed,

            q_g = ||grad_g||_2 / w_g

        is therefore the lambda scale at which g would first become KKT
        admissible for entry. v3.6 uses max_g q_g as a data-driven guide for
        the next continuation point. This is an event predictor rather than
        an exact knot formula because the residual changes along the path.
        """
        features_by_output = self._working_indices_by_output(working_set)
        residual = self._predict_restricted(Theta, B, features_by_output) - Y
        gradient = self._full_gradient(Theta, residual)
        flat = gradient.ravel(order="C")

        working = set(working_set)
        q_max = 0.0
        support_max = None
        for s, idx in self.groups.items():
            if s in working:
                continue
            w = float(weights[s])
            if w <= 0.0 or not np.isfinite(w):
                continue
            q = float(np.linalg.norm(flat[idx]) / w)
            if q > q_max:
                q_max = q
                support_max = s

        return float(q_max), support_max

    def _omitted_entry_score_from_residual(
        self,
        Theta: Array,
        residual: Array,
        weights,
        supports,
    ):
        """Largest omitted-group KKT entry score for an arbitrary residual.

        This is the common scale used by v3.6 to compare unresolved structure
        in the central target against structure that can be induced purely by
        the estimated finite-resolution target error.
        """
        gradient = self._full_gradient(Theta, residual)
        flat = gradient.ravel(order="C")
        active = set(supports)
        q_max = 0.0
        support_max = None
        for s, idx in self.groups.items():
            if s in active:
                continue
            w = float(weights[s])
            if w <= 0.0 or not np.isfinite(w):
                continue
            q = float(np.linalg.norm(flat[idx]) / w)
            if q > q_max:
                q_max = q
                support_max = s
        return float(q_max), support_max

    def _target_uncertainty_kkt_certificate(
        self,
        Theta: Array,
        Y: Array,
        B_refit: Array,
        supports,
        weights,
        uncertainty_fields,
    ):
        """Support-conditioned data-derived stopping certificate.

        Let r_S be the residual of the unpenalized refit of the central target
        on the current structural support S.  Its strongest omitted-group KKT
        score q_data measures the next unresolved structural event.

        Each uncertainty field U_l is the difference between the frozen
        central F1 target and one higher-order local epsilon extrapolant.  U_l
        is first projected onto S; only the part not representable by S is
        retained.  The strongest omitted-group KKT score of that residual is
        q_l.  Two consecutive extrapolation levels are used.  Their maximum
        plus their convergence spread defines a conservative, fully
        data-derived uncertainty entry floor

            q_floor = max(q_l) + (max(q_l) - min(q_l)).

        The support is certified when q_data <= q_floor: the next event is no
        stronger than an event that the target-construction uncertainty can
        itself generate.  No oracle support, fixed lambda_min, or hand-tuned
        MSE threshold enters this test.
        """
        if not uncertainty_fields:
            return {
                "uncertainty_certified": False,
                "unresolved_kkt_score": np.nan,
                "unresolved_kkt_support": None,
                "uncertainty_kkt_floor": np.nan,
                "uncertainty_kkt_ratio": np.nan,
                "uncertainty_kkt_levels": tuple(),
            }

        residual_data = Theta @ B_refit - Y
        q_data, q_data_support = self._omitted_entry_score_from_residual(
            Theta, residual_data, weights, supports
        )

        q_levels = []
        for U in uncertainty_fields:
            U = np.asarray(U, dtype=float)
            B_u = self._refit(Theta, U, supports)
            residual_u = Theta @ B_u - U
            q_u, _ = self._omitted_entry_score_from_residual(
                Theta, residual_u, weights, supports
            )
            q_levels.append(float(q_u))

        q_levels_arr = np.asarray(q_levels, dtype=float)
        q_hi = float(np.max(q_levels_arr)) if len(q_levels_arr) else 0.0
        q_lo = float(np.min(q_levels_arr)) if len(q_levels_arr) else 0.0
        q_spread = max(0.0, q_hi - q_lo)
        q_floor = float(q_hi + q_spread)

        tiny = np.finfo(float).tiny
        if q_floor <= tiny:
            ratio = 0.0 if q_data <= tiny else np.inf
        else:
            ratio = float(q_data / q_floor)

        return {
            "uncertainty_certified": bool(q_data <= q_floor),
            "unresolved_kkt_score": float(q_data),
            "unresolved_kkt_support": q_data_support,
            "uncertainty_kkt_floor": float(q_floor),
            "uncertainty_kkt_ratio": float(ratio),
            "uncertainty_kkt_levels": tuple(float(x) for x in q_levels),
        }

    def _adaptive_next_lambda(
        self,
        lam: float,
        lambda_max: float,
        q_max: float,
    ) -> float:
        """Choose the next decreasing lambda from KKT entry information.

        Two controls are combined:

        * geometric safeguard: at least a conventional continuation step;
        * KKT event jump: if the next omitted-group event is far away, jump
          close to it instead of densely traversing a support plateau.

        A maximum jump in log10(lambda) prevents a noisy/tiny q_max from
        sending the path directly to the emergency safety floor.
        """
        lam = float(lam)
        safety = float(lambda_max * self.lambda_min_ratio)
        geometric = float(self.adaptive_geometric_ratio * lam)

        if np.isfinite(q_max) and q_max > 0.0:
            event = float(self.adaptive_entry_fraction * q_max)
            candidate = min(geometric, event)
        else:
            candidate = geometric

        max_jump_floor = float(
            lam * (10.0 ** (-self.adaptive_max_jump_decades))
        )
        candidate = max(candidate, max_jump_floor, safety)

        # Numerical guard: continuation must strictly decrease lambda.
        if candidate >= lam * (1.0 - 1e-12):
            candidate = max(geometric, safety)

        return float(candidate)

    def _solve_one_lambda(self, Theta, Y, lam, weights, B_start, step_size, working_set):
        working = set(working_set)
        B = B_start.copy()
        total_iterations = 0
        all_converged = True
        final_restricted_converged = False
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
            final_restricted_converged = bool(info["converged"])
            all_converged = all_converged and final_restricted_converged
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
            # `converged` refers to the FINAL restricted solve only.
            # Intermediate working-set solves are tracked separately.
            "converged": final_restricted_converged,
            "all_intermediate_converged": all_converged,
            "relative_change": final_relative_change,
            "kkt_expansions": expansions,
            "kkt_reactivations": total_reactivated,
            "working_set_size": len(working),
            "max_kkt_ratio": final_max_kkt_ratio,
            "final_kkt_satisfied": (
                final_max_kkt_ratio <= 1.0 + self.kkt_tol
            ),
            "kkt_certified": (
                final_restricted_converged
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

    def _support_feature_counts(self, supports, n_outputs: int):
        """Number of unique active polynomial features for each output."""
        if not supports:
            return np.zeros(n_outputs, dtype=int)
        features_by_output = self._working_indices_by_output(supports)
        return np.asarray([len(features) for features in features_by_output], dtype=int)

    def _refit_identifiable(self, Theta: Array, Y: Array, supports):
        """
        Guard the post-selection OLS refit against saturated output designs.

        The unpenalized refit is identifiable only when every output has fewer
        active scalar directions than fit samples. Entering p_i >= n_fit is a
        hard stop: such candidates are not used for validation selection and
        are never passed to np.linalg.lstsq.
        """
        counts = self._support_feature_counts(supports, Y.shape[1])
        max_count = int(counts.max()) if len(counts) else 0
        return bool(np.all(counts < Theta.shape[0])), max_count, counts

    def _refit(self, Theta: Array, Y: Array, supports):
        """Unpenalized post-selection refit, guarded against saturation."""
        identifiable, max_count, _ = self._refit_identifiable(Theta, Y, supports)
        if not identifiable:
            raise RuntimeError(
                f"{self.label}: saturated post-selection refit requested "
                f"(max active features/output={max_count}, "
                f"n_samples={Theta.shape[0]})."
            )

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
        """Training diagnostics. BIC is retained for reporting only."""
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

    @staticmethod
    def _validation_stats(Theta: Array, Y: Array, B: Array):
        """
        Held-out validation loss with a state-level standard error.

        Each validation state contributes one loss value: mean squared field
        error across outputs. The standard error is therefore computed across
        states rather than across the flattened state-output matrix.
        """
        pred = Theta @ B
        residual = Y - pred
        state_mse = np.mean(residual ** 2, axis=1)
        mse = float(np.mean(state_mse))
        if len(state_mse) > 1:
            se = float(np.std(state_mse, ddof=1) / np.sqrt(len(state_mse)))
        else:
            se = 0.0
        return {
            "validation_mse": mse,
            "validation_se": se,
            "validation_relative_error": _relative_field_error(pred, Y),
        }

    def _fit_validation_split(self, Theta: Array, Y: Array):
        """Deterministic internal split used only for lambda selection."""
        n = Theta.shape[0]
        if n < 10:
            raise ValueError(
                f"{self.label}: at least 10 training samples are required for "
                "the internal validation split."
            )
        n_val = max(1, int(round(self.validation_fraction * n)))
        n_val = min(n_val, n - 1)
        rng = np.random.default_rng(self.validation_seed)
        permutation = rng.permutation(n)
        val_idx = np.sort(permutation[:n_val])
        fit_idx = np.sort(permutation[n_val:])
        return (
            Theta[fit_idx],
            Y[fit_idx],
            Theta[val_idx],
            Y[val_idx],
            fit_idx,
            val_idx,
        )

    def _early_stop_reason(self, coarse_rows):
        """Validation-aware path stopping rule for decreasing lambda."""
        if len(coarse_rows) == 0:
            return None

        last = coarse_rows[-1]
        if not bool(last.get("refit_identifiable", True)):
            return "post-selection refit became saturated"

        # Exact held-out fit is itself a safe stopping certificate for the
        # essentially noiseless synthetic benchmarks. Unlike v3.4, v3.5 does
        # not require several repeated supports after reaching the numerical
        # floor; that would risk an unnecessary KKT jump to a far smaller
        # lambda. Noisy empirical data will never satisfy this threshold.
        if (
            last["n_groups"] > 0
            and np.isfinite(last["validation_relative_error"])
            and last["validation_relative_error"] <= self.exact_fit_relative_error
        ):
            return "validation reached exact-fit numerical floor"

        if len(coarse_rows) < self.early_stop_min_evals:
            return None

        valid = [
            row for row in coarse_rows
            if bool(row.get("refit_identifiable", False))
            and np.isfinite(row.get("validation_mse", np.inf))
        ]
        if len(valid) < self.early_stop_min_evals:
            return None

        best = min(valid, key=lambda row: row["validation_mse"])
        best_pos = coarse_rows.index(best)
        after_best = coarse_rows[best_pos + 1:]
        if len(after_best) < self.early_stop_patience:
            return None

        threshold = (
            best["validation_mse"]
            * (1.0 + self.early_stop_relative_degradation)
            + best["validation_se"]
        )
        group_threshold = max(
            best["n_groups"] + 1,
            int(np.ceil(best["n_groups"] * self.early_stop_support_growth_ratio)),
        )
        tail = after_best[-self.early_stop_patience:]
        degraded = all(row["validation_mse"] > threshold for row in tail)
        grown = all(row["n_groups"] >= group_threshold for row in tail)
        if degraded and grown:
            return (
                "validation degraded while support kept growing "
                f"for {self.early_stop_patience} consecutive lambdas"
            )

        return None

    @staticmethod
    def _one_se_select(ledger: pd.DataFrame):
        """
        Select the largest lambda inside the one-standard-error envelope.

        This is the sparse analogue of the conventional CV 1-SE rule. The
        external test set is not involved.
        """
        eligible = ledger[
            ledger["refit_identifiable"].astype(bool)
            & np.isfinite(ledger["validation_mse"].to_numpy(dtype=float))
        ].copy()
        if len(eligible) == 0:
            raise RuntimeError("No identifiable validation candidate is available.")

        min_idx = eligible["validation_mse"].idxmin()
        min_row = eligible.loc[min_idx]
        threshold = float(min_row["validation_mse"] + min_row["validation_se"])
        envelope = eligible[eligible["validation_mse"] <= threshold].copy()
        # Largest lambda = strongest regularization = sparsest admissible point.
        best_idx = envelope["lambda"].idxmax()
        return best_idx, min_idx, threshold


    def _group_effect_scores(self, Theta: Array, B: Array, supports):
        """
        Score structural groups by their fitted field contribution.

        For group S,

            score(S) = ||Phi_S beta_S||_F / sqrt(n_states).

        Unlike a raw coefficient norm, this incorporates feature scaling,
        within-group basis multiplicity, and the sampled state distribution.
        Scores are computed only on the internal FIT split; validation remains
        untouched for pruning-model selection.
        """
        Theta = np.asarray(Theta, dtype=float)
        B = np.asarray(B, dtype=float)
        n_states = max(int(Theta.shape[0]), 1)
        n_outputs = B.shape[1]
        scores = {}

        for support in supports:
            idx = self.groups[support]
            outputs = idx % n_outputs
            features = idx // n_outputs
            norm_sq = 0.0
            for output in np.unique(outputs):
                mask = outputs == output
                feat = features[mask]
                if len(feat) == 0:
                    continue
                contribution = Theta[:, feat] @ B[feat, output]
                norm_sq += float(np.dot(contribution, contribution))
            scores[support] = float(np.sqrt(norm_sq / n_states))

        return scores

    def _prune_screened_support(
        self,
        Theta_fit: Array,
        Y_fit: Array,
        Theta_val: Array,
        Y_val: Array,
        screening_supports,
    ):
        """
        Contribution-ranked post-screening sparsification.

        AGLASSO supplies a high-recall candidate support. The candidate model
        is OLS-refit on the internal fit split, groups are ranked by fitted field
        contribution, and a deterministic nested pruning path is evaluated on
        validation data. The final support is the SMALLEST model whose
        validation MSE lies within

            best_MSE * (1 + pruning_relative_tolerance) + best_SE.

        No oracle support, external test data, or hard coefficient threshold is
        used.
        """
        screening_supports = tuple(
            sorted(screening_supports, key=lambda s: (len(s), s))
        )
        if not screening_supports:
            empty = pd.DataFrame(columns=[
                "prune_index", "n_groups", "threshold", "validation_mse",
                "validation_se", "validation_relative_error", "supports",
            ])
            return screening_supports, empty, np.nan, {}, np.nan

        B_screen = self._refit(Theta_fit, Y_fit, screening_supports)
        scores = self._group_effect_scores(Theta_fit, B_screen, screening_supports)

        ranked = sorted(
            screening_supports,
            key=lambda s: (-scores[s], len(s), s),
        )
        n_groups = len(ranked)
        min_keep = max(1, int(np.ceil(n_groups * self.pruning_min_keep_fraction)))

        raw_counts = np.linspace(
            n_groups,
            min_keep,
            num=min(self.pruning_n_candidates, n_groups),
        )
        keep_counts = sorted(
            set(int(round(x)) for x in raw_counts) | {n_groups, min_keep},
            reverse=True,
        )

        rows = []
        for prune_index, keep_count in enumerate(keep_counts):
            retained_ranked = ranked[:keep_count]
            retained = tuple(sorted(retained_ranked, key=lambda s: (len(s), s)))
            identifiable, max_active_features, _ = self._refit_identifiable(
                Theta_fit, Y_fit, retained
            )
            if identifiable:
                B_tmp = self._refit(Theta_fit, Y_fit, retained)
                vstats = self._validation_stats(Theta_val, Y_val, B_tmp)
                train_stats = self._stats(Theta_fit, Y_fit, B_tmp, retained)
            else:
                vstats = {
                    "validation_mse": np.inf,
                    "validation_se": np.inf,
                    "validation_relative_error": np.inf,
                }
                train_stats = {
                    "rss": np.inf,
                    "bic": np.inf,
                    "n_parameters": int(sum(len(self.groups[s]) for s in retained)),
                    "relative_error": np.inf,
                }

            threshold = float(min(scores[s] for s in retained_ranked))
            rows.append({
                "prune_index": int(prune_index),
                "n_groups": int(len(retained)),
                "threshold": threshold,
                "train_relative_error": float(train_stats["relative_error"]),
                "validation_relative_error": float(
                    vstats["validation_relative_error"]
                ),
                "validation_mse": float(vstats["validation_mse"]),
                "validation_se": float(vstats["validation_se"]),
                "refit_identifiable": bool(identifiable),
                "max_active_features_per_output": int(max_active_features),
                "n_parameters": int(train_stats["n_parameters"]),
                "supports": retained,
            })

        ledger = pd.DataFrame(rows)
        eligible = ledger[
            ledger["refit_identifiable"].astype(bool)
            & np.isfinite(ledger["validation_mse"].to_numpy(dtype=float))
        ].copy()
        if len(eligible) == 0:
            raise RuntimeError(f"{self.label}: no identifiable pruning candidate.")

        min_idx = eligible["validation_mse"].idxmin()
        min_row = eligible.loc[min_idx]
        tolerance = float(
            min_row["validation_mse"] * (1.0 + self.pruning_relative_tolerance)
            + min_row["validation_se"]
        )
        admissible = eligible[eligible["validation_mse"] <= tolerance].copy()

        # Primary objective: smallest support. Secondary: lower validation loss.
        # Tertiary: higher effect threshold (more conservative pruning).
        admissible = admissible.sort_values(
            ["n_groups", "validation_mse", "threshold"],
            ascending=[True, True, False],
        )
        chosen = admissible.iloc[0]
        selected_supports = tuple(chosen["supports"])
        selected_threshold = float(chosen["threshold"])

        return (
            selected_supports,
            ledger,
            selected_threshold,
            scores,
            tolerance,
            chosen.to_dict(),
        )

    def fit(
        self,
        Theta_train,
        Y_train,
        Theta_test=None,
        Y_test=None,
        *,
        target_uncertainty_fields: Optional[Sequence[Array]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_fingerprint: Optional[str] = None,
        resume_from_checkpoint: bool = True,
    ):
        """
        Fit one generator with validation-selected Adaptive Group LASSO.

        v3.6 keeps the v3.5 KKT-adaptive path and adds a support-conditioned
        target-uncertainty KKT stopping certificate.

        1. The provided training states are split deterministically into an
           internal fit set and held-out validation set.
        2. The path starts at data-derived lambda_max. After each solution,
           omitted-group KKT entry scores estimate where the next structural
           event lies; the next lambda is chosen from those scores plus a
           geometric safeguard.
        3. Post-selection OLS is used only for identifiable supports
           (p_i < n_fit for every output). Validation loss is measured on the
           held-out split. BIC is diagnostic only.
        4. When target_uncertainty_fields are supplied, the strongest
           unresolved omitted-group KKT event is compared with the
           data-derived uncertainty-induced KKT floor. The path stops as soon
           as the next event is no longer distinguishable from target error.
           Existing validation/numerical/safety stops remain fallbacks.
        5. Optional local refinement is centered on an interior minimum
           validation-loss adaptive point, never on a BIC minimum.
        6. If an uncertainty-certified support is found, it is selected
           directly. Otherwise the v3.5 minimum-validation screening plus
           contribution-pruning fallback is used.
        7. The final support is refit on the *full* training set; the external
           test set is used exactly once for final evaluation.
        """
        Theta_train = np.asarray(Theta_train, dtype=float)
        Y_train = np.asarray(Y_train, dtype=float)
        if Theta_test is not None:
            Theta_test = np.asarray(Theta_test, dtype=float)
        if Y_test is not None:
            Y_test = np.asarray(Y_test, dtype=float)

        if target_uncertainty_fields is None:
            uncertainty_train_fields = tuple()
        else:
            uncertainty_train_fields = tuple(
                np.asarray(U, dtype=float) for U in target_uncertainty_fields
            )
            for U in uncertainty_train_fields:
                if U.shape != Y_train.shape:
                    raise ValueError(
                        f"{self.label}: every target uncertainty field must "
                        f"match Y_train shape {Y_train.shape}; got {U.shape}."
                    )

        (
            Theta_fit,
            Y_fit,
            Theta_val,
            Y_val,
            fit_idx,
            val_idx,
        ) = self._fit_validation_split(Theta_train, Y_train)

        uncertainty_fit_fields = tuple(
            U[fit_idx] for U in uncertainty_train_fields
        )

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_fingerprint is None:
                raise ValueError(
                    "checkpoint_fingerprint is required when checkpoint_path is set."
                )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if not resume_from_checkpoint and checkpoint_path.exists():
                checkpoint_path.unlink()

        matrix_shape = (Theta_fit.shape[1], Y_fit.shape[1])
        group_keys = tuple(self.groups.keys())

        def pack_matrix(B: Array):
            flat = np.asarray(B, dtype=float).ravel(order="C")
            nz = np.flatnonzero(flat != 0.0)
            return {
                "nz_idx": nz.astype(np.int64, copy=True),
                "nz_values": flat[nz].copy(),
            }

        def unpack_matrix(state) -> Array:
            B = np.zeros(matrix_shape, dtype=float)
            if state is None:
                return B
            idx = np.asarray(state.get("nz_idx", ()), dtype=np.int64)
            values = np.asarray(state.get("nz_values", ()), dtype=float)
            if len(idx):
                flat = B.ravel(order="C")
                flat[idx] = values
                B = flat.reshape(matrix_shape, order="C")
            return B

        def save_partial(state) -> None:
            if checkpoint_path is None:
                return
            payload = {
                "implementation_version": _IMPLEMENTATION_VERSION,
                "checkpoint_format_version": _F1_PARTIAL_CHECKPOINT_FORMAT_VERSION,
                "fingerprint": checkpoint_fingerprint,
                "label": self.label,
                "state": state,
            }
            tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            with gzip.open(tmp, "wb", compresslevel=3) as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, checkpoint_path)

        def load_partial():
            if (
                checkpoint_path is None
                or not resume_from_checkpoint
                or not checkpoint_path.exists()
            ):
                return None
            try:
                with gzip.open(checkpoint_path, "rb") as fh:
                    payload = pickle.load(fh)
            except Exception as exc:
                if self.verbose:
                    print(f"  Partial path checkpoint unreadable; ignoring: {exc}")
                return None

            if payload.get("implementation_version") != _IMPLEMENTATION_VERSION:
                if self.verbose:
                    print(
                        "  Partial path checkpoint version mismatch; ignoring "
                        f"({payload.get('implementation_version')} != "
                        f"{_IMPLEMENTATION_VERSION})."
                    )
                return None
            if (
                payload.get("checkpoint_format_version")
                != _F1_PARTIAL_CHECKPOINT_FORMAT_VERSION
            ):
                if self.verbose:
                    print("  Partial path checkpoint format mismatch; ignoring.")
                return None
            if payload.get("fingerprint") != checkpoint_fingerprint:
                if self.verbose:
                    print(
                        "  Partial path checkpoint fingerprint mismatch; "
                        "ignoring stale checkpoint."
                    )
                return None

            state = payload.get("state")
            if not isinstance(state, dict):
                if self.verbose:
                    print("  Partial path checkpoint payload is invalid; ignoring.")
                return None
            if self.verbose:
                print(f"  Partial path checkpoint restored: {checkpoint_path.resolve()}")
                print(
                    "  Resume state: "
                    f"stage={state.get('stage')} | "
                    f"next_index={state.get('next_index')} | "
                    f"completed_lambdas={len(state.get('rows', []))}"
                )
            return state

        def rebuild_stats_cache(rows):
            cache = {}
            for row in rows:
                supports = tuple(row["active_supports"])
                cache[supports] = {
                    "rss": float(row["rss"]),
                    "bic": float(row["bic"]),
                    "n_parameters": int(row["n_parameters"]),
                    "relative_error": float(row["train_relative_error"]),
                    "validation_mse": float(row["validation_mse"]),
                    "validation_se": float(row["validation_se"]),
                    "validation_relative_error": float(row["validation_relative_error"]),
                    "refit_identifiable": bool(row["refit_identifiable"]),
                    "max_active_features_per_output": int(
                        row["max_active_features_per_output"]
                    ),
                    "uncertainty_certified": bool(
                        row.get("uncertainty_certified", False)
                    ),
                    "unresolved_kkt_score": float(
                        row.get("unresolved_kkt_score", np.nan)
                    ),
                    "unresolved_kkt_support": row.get(
                        "unresolved_kkt_support", None
                    ),
                    "uncertainty_kkt_floor": float(
                        row.get("uncertainty_kkt_floor", np.nan)
                    ),
                    "uncertainty_kkt_ratio": float(
                        row.get("uncertainty_kkt_ratio", np.nan)
                    ),
                    "uncertainty_kkt_levels": tuple(
                        row.get("uncertainty_kkt_levels", tuple())
                    ),
                }
            return cache

        resume_state = load_partial()

        if resume_state is None:
            B_init, ridge_alphas = self._initial_ridge(Theta_fit, Y_fit)
            weights = self._adaptive_weights(B_init)
            lambda_max = self._lambda_max(Theta_fit, Y_fit, weights)
            if lambda_max <= 0:
                raise RuntimeError(f"{self.label}: lambda_max is non-positive.")

            lipschitz = self._power_largest_eigenvalue_design(Theta_fit)
            if lipschitz <= 0.0:
                raise RuntimeError(f"{self.label}: non-positive Lipschitz constant.")
            step_size = 1.0 / lipschitz

            finite_alphas = ridge_alphas[np.isfinite(ridge_alphas)]
            alpha_min = float(np.min(finite_alphas)) if len(finite_alphas) else np.nan
            alpha_max = float(np.max(finite_alphas)) if len(finite_alphas) else np.nan
            weight_values = np.asarray([weights[s] for s in group_keys], dtype=float)

            rows = []
            coarse_states = []
            total_reactivations = 0
            evaluation_index = 0
            stats_cache = {}
            resume_stage = "adaptive"
            resume_next_index = 0
            resumed_next_lambda = None
            resumed_B_state = None
            resumed_working_set = tuple()
            coarse_stop_reason = ""
        else:
            weight_values = np.asarray(resume_state["weight_values"], dtype=float)
            if len(weight_values) != len(group_keys):
                raise RuntimeError(
                    f"{self.label}: checkpoint group universe has incompatible size."
                )
            weights = {s: float(w) for s, w in zip(group_keys, weight_values)}
            lambda_max = float(resume_state["lambda_max"])
            lipschitz = float(resume_state["lipschitz"])
            step_size = 1.0 / lipschitz
            alpha_min = float(resume_state["alpha_min"])
            alpha_max = float(resume_state["alpha_max"])
            rows = list(resume_state.get("rows", []))
            coarse_states = list(resume_state.get("coarse_states", []))
            total_reactivations = int(resume_state.get("total_reactivations", 0))
            evaluation_index = int(resume_state.get("evaluation_index", len(rows)))
            stats_cache = rebuild_stats_cache(rows)
            resume_stage = str(resume_state.get("stage", "adaptive"))
            resume_next_index = int(resume_state.get("next_index", 0))
            resumed_next_lambda = resume_state.get("next_lambda")
            if resumed_next_lambda is not None:
                resumed_next_lambda = float(resumed_next_lambda)
            resumed_B_state = resume_state.get("B_state")
            resumed_working_set = tuple(resume_state.get("working_set", tuple()))
            coarse_stop_reason = str(resume_state.get("coarse_stop_reason", ""))

        safety_lambda = float(lambda_max * self.lambda_min_ratio)
        start_time = time.perf_counter()

        if self.verbose:
            print(
                f"Starting KKT working-set Adaptive Group LASSO for {self.label} "
                "(KKT-event-driven validation path)..."
            )
            print(
                f"  internal split = {Theta_fit.shape[0]} fit / "
                f"{Theta_val.shape[0]} validation states"
            )
            print(
                f"  adaptive path <= {self.adaptive_max_evals} lambda evaluations | "
                f"geometric ratio={self.adaptive_geometric_ratio:.3f} | "
                f"entry fraction={self.adaptive_entry_fraction:.3f}"
            )
            print(
                f"  maximum KKT jump = {self.adaptive_max_jump_decades:.2f} decades | "
                f"emergency floor = {self.lambda_min_ratio:.0e} * lambda_max"
            )
            print(
                f"  local refinement <= {self.refine_n_lambdas} lambdas "
                "around an interior minimum validation-loss adaptive point"
            )
            if uncertainty_fit_fields:
                print(
                    "  selector = data-derived target-uncertainty KKT floor "
                    "(validation/pruning fallback if uncertified)"
                )
                print(
                    f"  target-uncertainty levels = {len(uncertainty_fit_fields)} "
                    "higher-order epsilon extrapolants"
                )
            elif self.enable_pruning:
                print(
                    "  selector = minimum-validation AGLASSO screening -> "
                    "contribution pruning -> sparsest validation-equivalent support"
                )
            else:
                print(
                    "  selector = validation one-standard-error rule "
                    "(largest admissible lambda)"
                )
            print(
                f"  pilot = adaptive Ridge | alpha range = "
                f"[{alpha_min:.3e}, {alpha_max:.3e}]"
            )
            if checkpoint_path is not None:
                print(f"  lambda-boundary checkpoint = {checkpoint_path.resolve()}")

        def evaluate_lambda(
            lam: float,
            *,
            B_start: Array,
            working_set,
            stage: str,
            stage_index: int,
        ):
            nonlocal total_reactivations, evaluation_index

            t0 = time.perf_counter()
            B_pen, working_out, info = self._solve_one_lambda(
                Theta_fit,
                Y_fit,
                float(lam),
                weights,
                B_start,
                step_size,
                working_set,
            )
            total_reactivations += int(info["kkt_reactivations"])
            supports = self._active_supports(B_pen, working_out)
            q_max, q_support = self._next_inactive_entry_score(
                Theta_fit,
                Y_fit,
                B_pen,
                weights,
                working_out,
            )
            adaptive_next_lambda = self._adaptive_next_lambda(
                float(lam),
                lambda_max,
                q_max,
            )

            if supports not in stats_cache:
                identifiable, max_active_features, _ = self._refit_identifiable(
                    Theta_fit, Y_fit, supports
                )
                if identifiable:
                    B_refit_tmp = self._refit(Theta_fit, Y_fit, supports)
                    stats = self._stats(Theta_fit, Y_fit, B_refit_tmp, supports)
                    stats.update(self._validation_stats(Theta_val, Y_val, B_refit_tmp))
                    stats.update(
                        self._target_uncertainty_kkt_certificate(
                            Theta_fit,
                            Y_fit,
                            B_refit_tmp,
                            supports,
                            weights,
                            uncertainty_fit_fields,
                        )
                    )
                    stats["refit_identifiable"] = True
                    stats["max_active_features_per_output"] = max_active_features
                else:
                    n_parameters = int(sum(len(self.groups[s]) for s in supports))
                    stats = {
                        "rss": np.inf,
                        "bic": np.inf,
                        "n_parameters": n_parameters,
                        "relative_error": np.inf,
                        "validation_mse": np.inf,
                        "validation_se": np.inf,
                        "validation_relative_error": np.inf,
                        "refit_identifiable": False,
                        "max_active_features_per_output": max_active_features,
                        "uncertainty_certified": False,
                        "unresolved_kkt_score": np.nan,
                        "unresolved_kkt_support": None,
                        "uncertainty_kkt_floor": np.nan,
                        "uncertainty_kkt_ratio": np.nan,
                        "uncertainty_kkt_levels": tuple(),
                    }
                stats_cache[supports] = stats
            stats = stats_cache[supports]

            row = {
                "path_index": evaluation_index,
                "evaluation_index": evaluation_index,
                "path_stage": stage,
                "stage_index": int(stage_index),
                "lambda": float(lam),
                "lambda_over_max": float(lam / lambda_max),
                "n_groups": len(supports),
                "n_parameters": stats["n_parameters"],
                "train_relative_error": stats["relative_error"],
                "validation_relative_error": stats["validation_relative_error"],
                "validation_mse": stats["validation_mse"],
                "validation_se": stats["validation_se"],
                "refit_identifiable": stats["refit_identifiable"],
                "max_active_features_per_output": stats[
                    "max_active_features_per_output"
                ],
                "rss": stats["rss"],
                "bic": stats["bic"],
                "solver_iterations": info["iterations"],
                "solver_converged": info["converged"],
                "all_intermediate_solver_runs_converged": info[
                    "all_intermediate_converged"
                ],
                "solver_relative_change": info["relative_change"],
                "working_set_size": info["working_set_size"],
                "kkt_expansions": info["kkt_expansions"],
                "kkt_reactivations": info["kkt_reactivations"],
                "max_kkt_ratio": info["max_kkt_ratio"],
                "final_kkt_satisfied": info["final_kkt_satisfied"],
                "kkt_certified": info["kkt_certified"],
                "next_entry_lambda": float(q_max),
                "next_entry_lambda_over_current": (
                    float(q_max / lam) if lam > 0.0 else np.nan
                ),
                "next_entry_support": q_support,
                "adaptive_next_lambda": float(adaptive_next_lambda),
                "adaptive_next_lambda_over_current": (
                    float(adaptive_next_lambda / lam) if lam > 0.0 else np.nan
                ),
                "uncertainty_certified": bool(
                    stats.get("uncertainty_certified", False)
                ),
                "unresolved_kkt_score": float(
                    stats.get("unresolved_kkt_score", np.nan)
                ),
                "unresolved_kkt_support": stats.get(
                    "unresolved_kkt_support", None
                ),
                "uncertainty_kkt_floor": float(
                    stats.get("uncertainty_kkt_floor", np.nan)
                ),
                "uncertainty_kkt_ratio": float(
                    stats.get("uncertainty_kkt_ratio", np.nan)
                ),
                "uncertainty_kkt_levels": tuple(
                    stats.get("uncertainty_kkt_levels", tuple())
                ),
                "lambda_runtime": float(time.perf_counter() - t0),
                "active_supports": supports,
            }
            rows.append(row)
            evaluation_index += 1
            return B_pen, working_out, row

        def checkpoint_state(
            *,
            stage: str,
            next_index: int,
            next_lambda: Optional[float],
            B_current: Array,
            working_set,
            stop_reason: str = "",
        ):
            save_partial({
                "stage": stage,
                "next_index": int(next_index),
                "next_lambda": (None if next_lambda is None else float(next_lambda)),
                "weight_values": weight_values,
                "lambda_max": float(lambda_max),
                "lipschitz": float(lipschitz),
                "alpha_min": float(alpha_min),
                "alpha_max": float(alpha_max),
                "B_state": pack_matrix(B_current),
                "working_set": tuple(working_set),
                "rows": rows,
                "coarse_states": coarse_states,
                "coarse_stop_reason": str(stop_reason),
                "total_reactivations": int(total_reactivations),
                "evaluation_index": int(evaluation_index),
            })

        # ------------------------------------------------------------
        # Stage 1: KKT-event-driven adaptive lambda continuation.
        # ------------------------------------------------------------
        if resume_stage == "adaptive":
            adaptive_index = resume_next_index
            B_warm = unpack_matrix(resumed_B_state)
            working_set = resumed_working_set
            if resumed_next_lambda is None:
                lam = float(lambda_max)
            else:
                lam = float(resumed_next_lambda)
        elif resume_stage == "refine":
            adaptive_index = len(coarse_states)
            B_warm = np.zeros(matrix_shape, dtype=float)
            working_set = tuple()
            lam = None
        else:
            raise RuntimeError(
                f"{self.label}: unknown checkpoint stage {resume_stage!r}."
            )

        if resume_stage == "adaptive":
            while adaptive_index < self.adaptive_max_evals:
                B_warm, working_set, row = evaluate_lambda(
                    float(lam),
                    B_start=B_warm,
                    working_set=working_set,
                    stage="adaptive",
                    stage_index=adaptive_index,
                )
                row["search_index"] = int(adaptive_index)

                flat = B_warm.ravel(order="C")
                nz = np.flatnonzero(flat != 0.0)
                coarse_states.append({
                    "lambda": float(lam),
                    "search_index": int(adaptive_index),
                    "path_stage": "adaptive",
                    "working_set": working_set,
                    "nz_idx": nz.astype(np.int64, copy=True),
                    "nz_values": flat[nz].copy(),
                    "validation_mse": float(row["validation_mse"]),
                })

                if self.verbose:
                    print(
                        f"  [adaptive {adaptive_index + 1:03d}/"
                        f"{self.adaptive_max_evals}] "
                        f"lambda={lam:.3e} | "
                        f"groups={row['n_groups']:5d} | "
                        f"WS={row['working_set_size']:6d} | "
                        f"val={row['validation_relative_error']:.3e} | "
                        f"entry/lambda={row['next_entry_lambda_over_current']:.3e} | "
                        f"next/lambda={row['adaptive_next_lambda_over_current']:.3e} | "
                        + (
                            f"uncert_ratio={row['uncertainty_kkt_ratio']:.3e} | "
                            if np.isfinite(row['uncertainty_kkt_ratio']) else ""
                        )
                        + f"KKT+={row['kkt_reactivations']:5d} | "
                        f"iter={row['solver_iterations']:5d} | "
                        f"elapsed={time.perf_counter() - start_time:.1f}s"
                    )

                search_rows = [
                    r for r in rows
                    if r["path_stage"] == "adaptive"
                ]
                reason = self._early_stop_reason(search_rows)

                if (
                    reason is None
                    and bool(row.get("uncertainty_certified", False))
                    and row["n_groups"] > 0
                ):
                    reason = (
                        "target-uncertainty KKT floor reached: next omitted "
                        f"event {row['unresolved_kkt_score']:.3e} <= "
                        f"uncertainty floor {row['uncertainty_kkt_floor']:.3e}"
                    )

                if reason is None and float(lam) <= safety_lambda * (1.0 + 1e-12):
                    reason = (
                        "emergency lambda safety floor reached "
                        f"({self.lambda_min_ratio:.0e} * lambda_max)"
                    )

                next_lam = float(row["adaptive_next_lambda"])
                if reason is None and next_lam >= float(lam) * (1.0 - 1e-12):
                    reason = "adaptive continuation could not decrease lambda"

                if reason is not None:
                    coarse_stop_reason = reason
                    checkpoint_state(
                        stage="refine",
                        next_index=0,
                        next_lambda=None,
                        B_current=B_warm,
                        working_set=working_set,
                        stop_reason=coarse_stop_reason,
                    )
                    if self.verbose:
                        print(f"  adaptive path stop: {reason}")
                    break

                checkpoint_state(
                    stage="adaptive",
                    next_index=adaptive_index + 1,
                    next_lambda=next_lam,
                    B_current=B_warm,
                    working_set=working_set,
                    stop_reason=coarse_stop_reason,
                )

                lam = next_lam
                adaptive_index += 1

            else:
                coarse_stop_reason = (
                    f"adaptive path reached max_evals={self.adaptive_max_evals}"
                )
                checkpoint_state(
                    stage="refine",
                    next_index=0,
                    next_lambda=None,
                    B_current=B_warm,
                    working_set=working_set,
                    stop_reason=coarse_stop_reason,
                )
                if self.verbose:
                    print(f"  adaptive path stop: {coarse_stop_reason}")

        if len(coarse_states) == 0:
            raise RuntimeError(f"{self.label}: no adaptive lambda was completed.")

        # ------------------------------------------------------------
        # Stage 2: optional validation-centered local refinement.
        # ------------------------------------------------------------
        coarse_rows_df = pd.DataFrame(
            [
                row for row in rows
                if row["path_stage"] == "adaptive"
            ]
        )
        eligible_coarse = coarse_rows_df[
            coarse_rows_df["refit_identifiable"].astype(bool)
            & np.isfinite(coarse_rows_df["validation_mse"].to_numpy(dtype=float))
        ]
        if len(eligible_coarse) == 0:
            raise RuntimeError(f"{self.label}: no identifiable search-path candidate.")

        coarse_best_label = eligible_coarse["validation_mse"].idxmin()
        coarse_best_search_index = int(
            coarse_rows_df.loc[coarse_best_label, "search_index"]
        )
        evaluated_search_indices = [
            int(state["search_index"]) for state in coarse_states
        ]
        # Position among actually evaluated adaptive search states.
        coarse_best_pos = evaluated_search_indices.index(coarse_best_search_index)

        refine_path = np.empty(0, dtype=float)
        hi_pos = lo_pos = None
        if (
            self.refine_n_lambdas > 0
            and "target-uncertainty KKT floor reached" not in coarse_stop_reason
            and 0 < coarse_best_pos < len(coarse_states) - 1
        ):
            hi_pos = coarse_best_pos - 1
            lo_pos = coarse_best_pos + 1
            lambda_hi = float(coarse_states[hi_pos]["lambda"])
            lambda_lo = float(coarse_states[lo_pos]["lambda"])
            refine_path = np.geomspace(
                lambda_hi,
                lambda_lo,
                self.refine_n_lambdas + 2,
            )[1:-1]

        if len(refine_path) > 0:
            if resume_stage == "refine" and resume_next_index > 0:
                refine_start_index = resume_next_index
                B_refine = unpack_matrix(resumed_B_state)
                working_refine = resumed_working_set
            else:
                # At refinement index 0 (including a resume immediately after
                # coarse early-stop), reconstruct the correct warm start from
                # the higher-lambda coarse neighbor rather than from the last
                # coarse point evaluated.
                refine_start_index = 0
                state_hi = coarse_states[hi_pos]
                B_refine = np.zeros(matrix_shape, dtype=float)
                flat_refine = B_refine.ravel(order="C")
                flat_refine[state_hi["nz_idx"]] = state_hi["nz_values"]
                B_refine = flat_refine.reshape(matrix_shape, order="C")
                working_refine = state_hi["working_set"]

            if self.verbose:
                print(
                    f"  refining around minimum search-path validation loss: "
                    f"lambda={coarse_states[coarse_best_pos]['lambda']:.3e}"
                )
                print(
                    f"  refinement bracket = "
                    f"[{coarse_states[hi_pos]['lambda']:.3e}, "
                    f"{coarse_states[lo_pos]['lambda']:.3e}] "
                    f"({len(refine_path)} new lambdas)"
                )

            progress_every_refine = max(1, len(refine_path) // 8)
            for refine_index in range(refine_start_index, len(refine_path)):
                lam = refine_path[refine_index]
                B_refine, working_refine, row = evaluate_lambda(
                    float(lam),
                    B_start=B_refine,
                    working_set=working_refine,
                    stage="refine",
                    stage_index=refine_index,
                )
                checkpoint_state(
                    stage="refine",
                    next_index=refine_index + 1,
                    next_lambda=None,
                    B_current=B_refine,
                    working_set=working_refine,
                    stop_reason=coarse_stop_reason,
                )

                if self.verbose and (
                    refine_index % progress_every_refine == 0
                    or refine_index == len(refine_path) - 1
                ):
                    print(
                        f"  [refine {refine_index + 1:03d}/{len(refine_path)}] "
                        f"lambda={lam:.3e} | groups={row['n_groups']:5d} | "
                        f"val={row['validation_relative_error']:.3e} | "
                        f"elapsed={time.perf_counter() - start_time:.1f}s"
                    )
        elif self.verbose and self.refine_n_lambdas > 0:
            print(
                "  refinement skipped: uncertainty certificate or search-path "
                "minimum lies on an evaluated boundary."
            )

        ledger = pd.DataFrame(rows)
        if len(ledger) == 0:
            raise RuntimeError(f"{self.label}: empty regularization-path ledger.")

        rank_order = np.argsort(-ledger["lambda"].to_numpy())
        lambda_rank = np.empty(len(ledger), dtype=int)
        lambda_rank[rank_order] = np.arange(len(ledger))
        ledger["lambda_rank"] = lambda_rank

        best_idx, min_val_idx, one_se_threshold = self._one_se_select(ledger)
        one_se = ledger.loc[best_idx]
        min_val = ledger.loc[min_val_idx]

        # Determine whether the minimum remains limited by the smallest
        # evaluated lambda after adaptive continuation/refinement.
        eligible = ledger[
            ledger["refit_identifiable"].astype(bool)
            & np.isfinite(ledger["validation_mse"].to_numpy(dtype=float))
        ]
        smallest_evaluated_lambda = float(eligible["lambda"].min())
        path_boundary_hit = bool(
            np.isclose(float(min_val["lambda"]), smallest_evaluated_lambda)
            and (
                "safety floor" in coarse_stop_reason
                or "max_evals" in coarse_stop_reason
            )
        )

        # ------------------------------------------------------------
        # Stage 3: uncertainty-certified support -> validation fallback.
        # ------------------------------------------------------------
        certified_rows = ledger[
            ledger["uncertainty_certified"].astype(bool)
            & ledger["refit_identifiable"].astype(bool)
        ] if "uncertainty_certified" in ledger.columns else ledger.iloc[0:0]

        if len(certified_rows) > 0:
            # Conservative structural choice: the largest lambda (hence first /
            # sparsest continuation point) for which no omitted event exceeds
            # the data-derived target-uncertainty KKT floor.
            cert_idx = certified_rows["lambda"].idxmax()
            screening_row = ledger.loc[cert_idx]
            screening_supports = tuple(screening_row["active_supports"])
            supports = screening_supports
            pruning_ledger = None
            pruning_threshold = np.nan
            pruning_scores = {}
            pruning_tolerance = np.nan
            validation_relative_error = float(
                screening_row["validation_relative_error"]
            )
            validation_mse = float(screening_row["validation_mse"])
            selection_method = "data-derived-target-uncertainty-KKT-floor"
            screening_lambda = float(screening_row["lambda"])
            screening_validation_relative_error = float(
                screening_row["validation_relative_error"]
            )
            kkt_row = screening_row

        elif self.enable_pruning:
            screening_row = min_val
            screening_supports = tuple(screening_row["active_supports"])
            (
                supports,
                pruning_ledger,
                pruning_threshold,
                pruning_scores,
                pruning_tolerance,
                pruning_choice,
            ) = self._prune_screened_support(
                Theta_fit,
                Y_fit,
                Theta_val,
                Y_val,
                screening_supports,
            )
            validation_relative_error = float(
                pruning_choice["validation_relative_error"]
            )
            validation_mse = float(pruning_choice["validation_mse"])
            selection_method = (
                "minimum-validation-AGLASSO-screening + "
                "contribution-pruning-validation"
            )
            screening_lambda = float(screening_row["lambda"])
            screening_validation_relative_error = float(
                screening_row["validation_relative_error"]
            )
            kkt_row = screening_row
        else:
            screening_row = one_se
            screening_supports = tuple(one_se["active_supports"])
            supports = screening_supports
            pruning_ledger = None
            pruning_threshold = np.nan
            pruning_scores = {}
            pruning_tolerance = np.nan
            validation_relative_error = float(one_se["validation_relative_error"])
            validation_mse = float(one_se["validation_mse"])
            selection_method = "internal-validation-1se-largest-lambda"
            screening_lambda = float(one_se["lambda"])
            screening_validation_relative_error = float(
                one_se["validation_relative_error"]
            )
            kkt_row = one_se

        # Final refit on all provided training states, after support selection.
        B_scaled = self._refit(Theta_train, Y_train, supports)
        B_raw = self.library.to_raw_coefficients(B_scaled)

        train_error = _relative_field_error(Theta_train @ B_scaled, Y_train)
        test_error = None
        if Theta_test is not None and Y_test is not None:
            test_error = _relative_field_error(Theta_test @ B_scaled, Y_test)

        counts = dict(sorted(Counter(len(s) for s in supports).items()))
        screening_counts = dict(
            sorted(Counter(len(s) for s in screening_supports).items())
        )
        selected_kkt = bool(kkt_row["kkt_certified"])
        all_path_kkt = bool(ledger["kkt_certified"].all())
        max_final_ratio = float(kkt_row["max_kkt_ratio"])
        final_restricted_converged = bool(kkt_row["solver_converged"])
        final_kkt_satisfied = bool(kkt_row["final_kkt_satisfied"])
        all_intermediate_converged = bool(
            kkt_row["all_intermediate_solver_runs_converged"]
        )

        if self.verbose:
            print(
                f"  minimum validation loss: lambda={min_val['lambda']:.3e} | "
                f"groups={int(min_val['n_groups'])} | "
                f"val_rel={min_val['validation_relative_error']:.3e}"
            )
            print(
                f"  one-SE threshold (MSE) = {one_se_threshold:.6e}"
            )
            if len(certified_rows) > 0:
                print(
                    f"  uncertainty-certified selection: lambda={screening_lambda:.3e} "
                    f"({screening_row['path_stage']}) | groups={len(supports)} | "
                    f"val_rel={validation_relative_error:.3e}"
                )
                print(
                    f"  unresolved KKT={screening_row['unresolved_kkt_score']:.3e} "
                    f"<= uncertainty floor={screening_row['uncertainty_kkt_floor']:.3e} "
                    f"| ratio={screening_row['uncertainty_kkt_ratio']:.3f}"
                )
            elif self.enable_pruning:
                print(
                    f"  AGLASSO screening: lambda={screening_lambda:.3e} "
                    f"({screening_row['path_stage']}) | "
                    f"groups={len(screening_supports)} | "
                    f"val_rel={screening_validation_relative_error:.3e}"
                )
                print(
                    f"  pruning selected {len(supports)}/{len(screening_supports)} "
                    f"groups | effect threshold={pruning_threshold:.3e} | "
                    f"val_rel={validation_relative_error:.3e} | "
                    f"admissible MSE ceiling={pruning_tolerance:.6e}"
                )
            else:
                print(
                    f"  selected lambda={one_se['lambda']:.3e} "
                    f"({one_se['path_stage']}) | groups={len(supports)} | "
                    f"val_rel={validation_relative_error:.3e} | "
                    "selector=largest lambda inside 1-SE envelope"
                )

            n_adaptive_eval = int((ledger["path_stage"] == "adaptive").sum())
            n_refine_eval = int((ledger["path_stage"] == "refine").sum())
            print(
                f"  evaluated {len(ledger)} lambdas total "
                f"({n_adaptive_eval} adaptive + {n_refine_eval} refine)"
            )
            if coarse_stop_reason:
                print(f"  search stop reason = {coarse_stop_reason}")
            if path_boundary_hit:
                print(
                    "  WARNING: minimum validation loss remains at the smallest "
                    "evaluated lambda because the adaptive path hit its emergency "
                    "limit."
                )

        result = GeneratorFitResult(
            coefficients_scaled=B_scaled,
            coefficients_raw=B_raw,
            selected_supports=supports,
            train_relative_error=train_error,
            test_relative_error=test_error,
            lambda_selected=screening_lambda,
            path_ledger=ledger,
            group_counts_by_size=counts,
            all_solver_runs_converged=bool(ledger["solver_converged"].all()),
            max_solver_iterations=int(ledger["solver_iterations"].max()),
            pilot_method="condition-number-adaptive Ridge",
            ridge_alpha_min=alpha_min,
            ridge_alpha_max=alpha_max,
            kkt_certified=selected_kkt,
            all_path_kkt_certified=all_path_kkt,
            validation_relative_error=validation_relative_error,
            validation_mse=validation_mse,
            selection_method=selection_method,
            n_fit_samples=int(len(fit_idx)),
            n_validation_samples=int(len(val_idx)),
            early_stop_reason=coarse_stop_reason,
            path_boundary_hit=path_boundary_hit,
            total_kkt_reactivations=total_reactivations,
            max_final_kkt_ratio=max_final_ratio,
            screening_supports=screening_supports,
            screening_lambda=screening_lambda,
            screening_validation_relative_error=screening_validation_relative_error,
            screening_group_counts_by_size=screening_counts,
            pruning_ledger=pruning_ledger,
            pruning_applied=bool(self.enable_pruning and len(certified_rows) == 0),
            pruning_threshold=float(pruning_threshold),
            pruning_score_method=(
                "fit-field-contribution"
                if (self.enable_pruning and len(certified_rows) == 0)
                else "none"
            ),
            final_restricted_converged=final_restricted_converged,
            final_kkt_satisfied=final_kkt_satisfied,
            all_intermediate_solver_runs_converged=all_intermediate_converged,
            uncertainty_certified=bool(
                kkt_row.get("uncertainty_certified", False)
            ),
            unresolved_kkt_score=float(
                kkt_row.get("unresolved_kkt_score", np.nan)
            ),
            uncertainty_kkt_floor=float(
                kkt_row.get("uncertainty_kkt_floor", np.nan)
            ),
            uncertainty_kkt_ratio=float(
                kkt_row.get("uncertainty_kkt_ratio", np.nan)
            ),
            uncertainty_kkt_levels=tuple(
                kkt_row.get("uncertainty_kkt_levels", tuple())
            ),
            uncertainty_next_support=kkt_row.get(
                "unresolved_kkt_support", None
            ),
        )

        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()
            if self.verbose:
                print("  Partial path checkpoint cleared after successful completion.")

        return result

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

    The remaining estimator settings are internal/frozen. v3.6 keeps the
    public modelling interface unchanged and retains the condition-adaptive
    Ridge pilot plus KKT working-set AGLASSO core. Both F^(0) and F^(1) use
    KKT-event-driven lambda continuation learned from the current residual and
    omitted structural groups. F^(1) additionally uses a data-derived
    extrapolation-uncertainty KKT floor to decide when the next structural
    event is no longer distinguishable from finite-resolution target error.
    Contribution-ranked validation pruning remains a fallback only.
    BIC remains diagnostic only.
    """

    _N_EXTRAPOLATION_POINTS = 4
    _EXTRAPOLATION_POLY_ORDER = 3

    # v3.6 data-derived F1 target-resolution estimator. These auxiliary
    # extrapolants are NOT alternative training targets and do not rerun the
    # sparse inference. They estimate the finite-epsilon error of the frozen
    # 4-point cubic target. Two consecutive higher-order local interpolants
    # provide both a floor and its one-step convergence spread.
    _UNCERTAINTY_AUX_EXTRAPOLATIONS = ((5, 4), (6, 5))

    _ADAPTIVE_GAMMA = 1.0
    _ADAPTIVE_DELTA_RATIO = 1e-8
    _MAX_ITER = 5000
    _SOLVER_TOL = 1e-8
    _ACTIVE_GROUP_TOL = 1e-10

    _RIDGE_CONDITION_TARGET = 1e6
    _RIDGE_ALPHA_FLOOR_RATIO = 1e-12
    _KKT_TOL = 1e-7
    _MAX_KKT_EXPANSIONS = 100

    # v3.6 KKT-event-driven lambda continuation. There is no benchmark-
    # specific nominal lambda_min. Both F0 and F1 start at data-derived
    # lambda_max and choose subsequent lambda values from omitted-group KKT
    # entry scores. The ratio below is only an emergency numerical floor.
    _F0_N_LAMBDAS = 2              # compatibility only; path is adaptive
    _F0_LAMBDA_MIN_RATIO = 1e-10    # emergency floor, not planned endpoint
    _F0_COARSE_N_LAMBDAS = 2        # compatibility only
    _F0_REFINE_N_LAMBDAS = 4

    _F1_N_LAMBDAS = 2              # compatibility only; path is adaptive
    _F1_LAMBDA_MIN_RATIO = 1e-10    # emergency floor, not planned endpoint
    _F1_COARSE_N_LAMBDAS = 2        # compatibility only
    _F1_REFINE_N_LAMBDAS = 8

    _ADAPTIVE_GEOMETRIC_RATIO = 0.75
    _ADAPTIVE_ENTRY_FRACTION = 0.98
    _ADAPTIVE_MAX_JUMP_DECADES = 2.0
    _F0_ADAPTIVE_MAX_EVALS = 40
    _F1_ADAPTIVE_MAX_EVALS = 60

    # F1 post-screening sparsification. Groups are ranked by fitted field
    # contribution on the internal fit split. Up to 40 nested supports are
    # validated, and the smallest model within 2% + 1 SE of the best pruning
    # validation MSE is selected. F0 pruning is disabled.
    _PRUNING_N_CANDIDATES = 40
    _PRUNING_MIN_KEEP_FRACTION = 0.05
    _PRUNING_RELATIVE_TOLERANCE = 0.02

    # Internal validation / path-stopping configuration. The external test set
    # is never used for lambda selection.
    _VALIDATION_FRACTION = 0.20
    _VALIDATION_SEED = 20260818
    _EARLY_STOP_MIN_EVALS = 8
    _EARLY_STOP_PATIENCE = 2
    _EARLY_STOP_RELATIVE_DEGRADATION = 0.02
    _EARLY_STOP_SUPPORT_GROWTH_RATIO = 1.10
    _EXACT_FIT_PLATEAU_PATIENCE = 3
    _EXACT_FIT_RELATIVE_ERROR = 1e-8

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
                "TSCInference v3.6 currently supports temporal_order=1 only "
                "(recovery of F^(0) and F^(1))."
            )

        self.library_: Optional[StructuralPolynomialLibrary] = None
        self.result_: Optional[TSCResult] = None

    @staticmethod
    def _update_hash_with_array(hasher, name: str, array: Optional[Array]):
        """Add an ndarray (including shape/dtype) to a reproducibility hash."""
        hasher.update(name.encode("utf-8"))
        if array is None:
            hasher.update(b"<NONE>")
            return
        arr = np.ascontiguousarray(np.asarray(array))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        hasher.update(arr.view(np.uint8).tobytes())

    def _f0_fingerprint(
        self,
        *,
        X0: Array,
        A_train: Array,
        eps_used: Array,
        X0_test: Optional[Array],
        A_test: Optional[Array],
    ) -> str:
        """
        Fingerprint every quantity that can change the completed F^(0) result.

        This prevents an F0 checkpoint created from a different dataset,
        extrapolation grid, library, or test split from being reused silently.
        """
        h = hashlib.sha256()
        h.update(_IMPLEMENTATION_VERSION.encode("utf-8"))
        config = (
            self.max_interaction_order,
            self.max_polynomial_degree,
            self.temporal_order,
            self._F0_N_LAMBDAS,
            self._F0_LAMBDA_MIN_RATIO,
            self._F0_COARSE_N_LAMBDAS,
            self._F0_REFINE_N_LAMBDAS,
            self._ADAPTIVE_GEOMETRIC_RATIO,
            self._ADAPTIVE_ENTRY_FRACTION,
            self._ADAPTIVE_MAX_JUMP_DECADES,
            self._F0_ADAPTIVE_MAX_EVALS,
            self._ADAPTIVE_GAMMA,
            self._ADAPTIVE_DELTA_RATIO,
            self._MAX_ITER,
            self._SOLVER_TOL,
            self._ACTIVE_GROUP_TOL,
            self._RIDGE_CONDITION_TARGET,
            self._RIDGE_ALPHA_FLOOR_RATIO,
            self._KKT_TOL,
            self._MAX_KKT_EXPANSIONS,
            self._VALIDATION_FRACTION,
            self._VALIDATION_SEED,
            self._EARLY_STOP_MIN_EVALS,
            self._EARLY_STOP_PATIENCE,
            self._EARLY_STOP_RELATIVE_DEGRADATION,
            self._EARLY_STOP_SUPPORT_GROWTH_RATIO,
            self._EXACT_FIT_PLATEAU_PATIENCE,
            self._EXACT_FIT_RELATIVE_ERROR,
        )
        h.update(repr(config).encode("utf-8"))
        self._update_hash_with_array(h, "X0", X0)
        self._update_hash_with_array(h, "A_train", A_train)
        self._update_hash_with_array(h, "eps_used", eps_used)
        self._update_hash_with_array(h, "X0_test", X0_test)
        self._update_hash_with_array(h, "A_test", A_test)
        return h.hexdigest()

    @staticmethod
    def _f0_checkpoint_path(checkpoint_dir: Union[str, Path]) -> Path:
        return Path(checkpoint_dir) / "TSC_F0_checkpoint.pkl.gz"

    def _load_f0_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        fingerprint: str,
    ) -> Optional[GeneratorFitResult]:
        path = self._f0_checkpoint_path(checkpoint_dir)
        if not path.exists():
            return None

        try:
            with gzip.open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            if self.verbose:
                print(f"  F0 checkpoint unreadable; ignoring: {exc}")
            return None

        if payload.get("implementation_version") != _IMPLEMENTATION_VERSION:
            if self.verbose:
                print(
                    "  F0 checkpoint version mismatch; ignoring "
                    f"({payload.get('implementation_version')} != "
                    f"{_IMPLEMENTATION_VERSION})."
                )
            return None

        if payload.get("fingerprint") != fingerprint:
            if self.verbose:
                print("  F0 checkpoint fingerprint mismatch; ignoring stale checkpoint.")
            return None

        F0 = payload.get("F0")
        if not isinstance(F0, GeneratorFitResult):
            if self.verbose:
                print("  F0 checkpoint payload is invalid; ignoring.")
            return None

        if self.verbose:
            print(f"  F0 checkpoint restored: {path.resolve()}")
        return F0

    def _save_f0_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        fingerprint: str,
        F0: GeneratorFitResult,
    ) -> Path:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = self._f0_checkpoint_path(directory)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "implementation_version": _IMPLEMENTATION_VERSION,
            "fingerprint": fingerprint,
            "F0": F0,
        }
        with gzip.open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        if self.verbose:
            print(f"  F0 checkpoint saved: {path.resolve()}")
        return path

    def _f1_fingerprint(
        self,
        *,
        X0: Array,
        F1_target_train: Array,
        F1_uncertainty_fields_train: Optional[Sequence[Array]],
        X0_test: Optional[Array],
        F1_target_test: Optional[Array],
    ) -> str:
        """
        Fingerprint every quantity that can change the F^(1) path.

        F1_target already contains the dependence on endpoint extrapolation,
        the completed F0 estimate, and the finite-flow correction. X0 fixes the
        scaled polynomial design matrix. The F1 path/solver configuration is
        included explicitly to prevent cross-run checkpoint reuse.
        """
        h = hashlib.sha256()
        h.update(_IMPLEMENTATION_VERSION.encode("utf-8"))
        config = (
            self.max_interaction_order,
            self.max_polynomial_degree,
            self.temporal_order,
            self._F1_N_LAMBDAS,
            self._F1_LAMBDA_MIN_RATIO,
            self._F1_COARSE_N_LAMBDAS,
            self._F1_REFINE_N_LAMBDAS,
            self._ADAPTIVE_GEOMETRIC_RATIO,
            self._ADAPTIVE_ENTRY_FRACTION,
            self._ADAPTIVE_MAX_JUMP_DECADES,
            self._F1_ADAPTIVE_MAX_EVALS,
            self._PRUNING_N_CANDIDATES,
            self._PRUNING_MIN_KEEP_FRACTION,
            self._PRUNING_RELATIVE_TOLERANCE,
            self._ADAPTIVE_GAMMA,
            self._ADAPTIVE_DELTA_RATIO,
            self._MAX_ITER,
            self._SOLVER_TOL,
            self._ACTIVE_GROUP_TOL,
            self._RIDGE_CONDITION_TARGET,
            self._RIDGE_ALPHA_FLOOR_RATIO,
            self._KKT_TOL,
            self._MAX_KKT_EXPANSIONS,
            self._VALIDATION_FRACTION,
            self._VALIDATION_SEED,
            self._EARLY_STOP_MIN_EVALS,
            self._EARLY_STOP_PATIENCE,
            self._EARLY_STOP_RELATIVE_DEGRADATION,
            self._EARLY_STOP_SUPPORT_GROWTH_RATIO,
            self._EXACT_FIT_PLATEAU_PATIENCE,
            self._EXACT_FIT_RELATIVE_ERROR,
            self._UNCERTAINTY_AUX_EXTRAPOLATIONS,
        )
        h.update(repr(config).encode("utf-8"))
        self._update_hash_with_array(h, "X0", X0)
        self._update_hash_with_array(h, "F1_target_train", F1_target_train)
        if F1_uncertainty_fields_train is None:
            h.update(b"<NO_F1_UNCERTAINTY_FIELDS>")
        else:
            for j, U in enumerate(F1_uncertainty_fields_train):
                self._update_hash_with_array(h, f"F1_uncertainty_{j}", U)
        self._update_hash_with_array(h, "X0_test", X0_test)
        self._update_hash_with_array(h, "F1_target_test", F1_target_test)
        return h.hexdigest()

    @staticmethod
    def _f1_partial_checkpoint_path(
        checkpoint_dir: Union[str, Path],
    ) -> Path:
        return Path(checkpoint_dir) / "TSC_F1_partial_checkpoint.pkl.gz"

    def fit(
        self,
        X0: Array,
        XF: Union[Array, Mapping[float, Array]],
        eps: Sequence[float],
        *,
        X0_test: Optional[Array] = None,
        XF_test: Optional[Union[Array, Mapping[float, Array]]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = True,
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

        # v3.6: estimate the finite-resolution uncertainty of the frozen
        # central G1 target from two consecutive higher-order local
        # extrapolants. These auxiliary fields are never used as training
        # targets and therefore do not multiply the sparse-inference cost.
        max_aux_points = max(
            n_points for n_points, _ in self._UNCERTAINTY_AUX_EXTRAPOLATIONS
        )
        G1_aux_train = []
        if len(eps) >= max_aux_points:
            for n_points, poly_order in self._UNCERTAINTY_AUX_EXTRAPOLATIONS:
                _, G1_aux, _ = _resolution_extrapolation(
                    X0,
                    XF_array,
                    eps,
                    n_points=n_points,
                    polynomial_order=poly_order,
                )
                G1_aux_train.append(G1_aux)
        elif self.verbose:
            print(
                "WARNING: fewer than "
                f"{max_aux_points} epsilon values are available; "
                "v3.6 target-uncertainty KKT certification is disabled "
                "and F1 will fall back to validation/pruning selection."
            )

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
            if G1_aux_train:
                print(
                    "  F1 uncertainty extrapolants = "
                    + ", ".join(
                        f"{n}pt/order{o}"
                        for n, o in self._UNCERTAINTY_AUX_EXTRAPOLATIONS
                    )
                )
            print(f"  library features/output = {library.n_features}")
            print(f"  structural groups = {len(library.structural_groups)}")
            print()

        # 3. F^(0)
        #
        # Optional completed-F0 checkpointing is deliberately placed *after*
        # extrapolation/library construction but *before* finite-flow/F1 work.
        # A data/config fingerprint prevents accidental reuse on another run.
        f0_fingerprint = self._f0_fingerprint(
            X0=X0,
            A_train=A_train,
            eps_used=eps_used,
            X0_test=X0_test if have_test else None,
            A_test=A_test,
        )

        F0 = None
        if checkpoint_dir is not None and resume_from_checkpoint:
            F0 = self._load_f0_checkpoint(
                checkpoint_dir,
                f0_fingerprint,
            )

        if F0 is None:
            estimator_F0 = _AdaptiveGroupLasso(
                library=library,
                n_lambdas=self._F0_N_LAMBDAS,
                lambda_min_ratio=self._F0_LAMBDA_MIN_RATIO,
                coarse_n_lambdas=self._F0_COARSE_N_LAMBDAS,
                refine_n_lambdas=self._F0_REFINE_N_LAMBDAS,
                gamma=self._ADAPTIVE_GAMMA,
                delta_ratio=self._ADAPTIVE_DELTA_RATIO,
                max_iter=self._MAX_ITER,
                tol=self._SOLVER_TOL,
                active_group_tol=self._ACTIVE_GROUP_TOL,
                ridge_condition_target=self._RIDGE_CONDITION_TARGET,
                ridge_alpha_floor_ratio=self._RIDGE_ALPHA_FLOOR_RATIO,
                kkt_tol=self._KKT_TOL,
                max_kkt_expansions=self._MAX_KKT_EXPANSIONS,
                validation_fraction=self._VALIDATION_FRACTION,
                validation_seed=self._VALIDATION_SEED,
                early_stop_min_evals=self._EARLY_STOP_MIN_EVALS,
                early_stop_patience=self._EARLY_STOP_PATIENCE,
                early_stop_relative_degradation=self._EARLY_STOP_RELATIVE_DEGRADATION,
                early_stop_support_growth_ratio=self._EARLY_STOP_SUPPORT_GROWTH_RATIO,
                exact_fit_plateau_patience=self._EXACT_FIT_PLATEAU_PATIENCE,
                exact_fit_relative_error=self._EXACT_FIT_RELATIVE_ERROR,
                adaptive_geometric_ratio=self._ADAPTIVE_GEOMETRIC_RATIO,
                adaptive_entry_fraction=self._ADAPTIVE_ENTRY_FRACTION,
                adaptive_max_jump_decades=self._ADAPTIVE_MAX_JUMP_DECADES,
                adaptive_max_evals=self._F0_ADAPTIVE_MAX_EVALS,
                verbose=self.verbose,
                label="F^(0)",
            )
            F0 = estimator_F0.fit(
                Theta_train,
                A_train,
                Theta_test,
                A_test,
            )

            if checkpoint_dir is not None:
                self._save_f0_checkpoint(
                    checkpoint_dir,
                    f0_fingerprint,
                    F0,
                )

        # 4. Finite-flow correction
        Q_train = _finite_flow_correction(X0, library, F0.coefficients_raw)
        F1_target_train = G1_train - Q_train

        if have_test:
            Q_test = _finite_flow_correction(X0_test, library, F0.coefficients_raw)
            F1_target_test = G1_test - Q_test
        else:
            Q_test = None
            F1_target_test = None

        # Target-resolution fields live directly on F1 because the same Q[F0]
        # is subtracted from both central and auxiliary G1 estimates. Hence
        # the finite-flow correction cancels exactly in the uncertainty field.
        F1_uncertainty_fields_train = tuple(
            G1_train - G1_aux for G1_aux in G1_aux_train
        )

        # 5. F^(1)
        estimator_F1 = _AdaptiveGroupLasso(
            library=library,
            n_lambdas=self._F1_N_LAMBDAS,
            lambda_min_ratio=self._F1_LAMBDA_MIN_RATIO,
            coarse_n_lambdas=self._F1_COARSE_N_LAMBDAS,
            refine_n_lambdas=self._F1_REFINE_N_LAMBDAS,
            gamma=self._ADAPTIVE_GAMMA,
            delta_ratio=self._ADAPTIVE_DELTA_RATIO,
            max_iter=self._MAX_ITER,
            tol=self._SOLVER_TOL,
            active_group_tol=self._ACTIVE_GROUP_TOL,
            ridge_condition_target=self._RIDGE_CONDITION_TARGET,
            ridge_alpha_floor_ratio=self._RIDGE_ALPHA_FLOOR_RATIO,
            kkt_tol=self._KKT_TOL,
            max_kkt_expansions=self._MAX_KKT_EXPANSIONS,
            validation_fraction=self._VALIDATION_FRACTION,
            validation_seed=self._VALIDATION_SEED,
            early_stop_min_evals=self._EARLY_STOP_MIN_EVALS,
            early_stop_patience=self._EARLY_STOP_PATIENCE,
            early_stop_relative_degradation=self._EARLY_STOP_RELATIVE_DEGRADATION,
            early_stop_support_growth_ratio=self._EARLY_STOP_SUPPORT_GROWTH_RATIO,
            exact_fit_plateau_patience=self._EXACT_FIT_PLATEAU_PATIENCE,
            exact_fit_relative_error=self._EXACT_FIT_RELATIVE_ERROR,
            adaptive_geometric_ratio=self._ADAPTIVE_GEOMETRIC_RATIO,
            adaptive_entry_fraction=self._ADAPTIVE_ENTRY_FRACTION,
            adaptive_max_jump_decades=self._ADAPTIVE_MAX_JUMP_DECADES,
            adaptive_max_evals=self._F1_ADAPTIVE_MAX_EVALS,
            enable_pruning=True,
            pruning_n_candidates=self._PRUNING_N_CANDIDATES,
            pruning_min_keep_fraction=self._PRUNING_MIN_KEEP_FRACTION,
            pruning_relative_tolerance=self._PRUNING_RELATIVE_TOLERANCE,
            verbose=self.verbose,
            label="F^(1)",
        )
        if checkpoint_dir is not None:
            f1_fingerprint = self._f1_fingerprint(
                X0=X0,
                F1_target_train=F1_target_train,
                F1_uncertainty_fields_train=F1_uncertainty_fields_train,
                X0_test=X0_test if have_test else None,
                F1_target_test=F1_target_test,
            )
            f1_partial_checkpoint = self._f1_partial_checkpoint_path(
                checkpoint_dir
            )
        else:
            f1_fingerprint = None
            f1_partial_checkpoint = None

        F1 = estimator_F1.fit(
            Theta_train,
            F1_target_train,
            Theta_test,
            F1_target_test,
            target_uncertainty_fields=F1_uncertainty_fields_train,
            checkpoint_path=f1_partial_checkpoint,
            checkpoint_fingerprint=f1_fingerprint,
            resume_from_checkpoint=resume_from_checkpoint,
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
        for j, U in enumerate(F1_uncertainty_fields_train, start=1):
            diagnostics[f"RMS(F1_target_uncertainty_{j})"] = _field_rms(U)

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
            F1_uncertainty_fields_train=F1_uncertainty_fields_train,
            diagnostics=diagnostics,
            _library=library,
        )
        self.result_ = result

        if self.verbose:
            print()
            result.summary(show_supports=False)

        return result


__all__ = ["TSCInference", "TSCResult", "GeneratorFitResult"]
