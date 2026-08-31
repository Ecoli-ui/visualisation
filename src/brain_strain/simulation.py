"""Reduced-order generalized-Maxwell response for the mesh viewer.

The constitutive model is a generalized Maxwell (Maxwell--Wiechert) solid,
written as an equilibrium spring in parallel with Maxwell branches::

    G(t) = G_inf + sum(G_i * exp(-t / tau_i)).

Each mesh cell is driven by the same mild-impact stress history, scaled by its
distance from a rotation axis.  This makes the material response and its
relaxation explicit, replacing the former analytic hotspot field.  It is
still a reduced-order demonstration: it does not enforce momentum balance,
contact, or boundary conditions and therefore is not a finite-element result.

The default sampling (18 ms) and reference maximum-shear-strain amplitudes
(0.017 for neck rotation and 0.011 for neck extension) follow Gomez et al.,
"Group characterization of impact-induced, in vivo human brain kinematics",
J. R. Soc. Interface 18 (2021), DOI 10.1098/rsif.2021.0251,
https://pmc.ncbi.nlm.nih.gov/articles/PMC8220272/.  That paper reports measured
kinematics, not Maxwell coefficients. The default Prony fractions and
relaxation times below are transparent numerical assumptions and must be
calibrated before physical use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pyvista as pv

ImpactMode = Literal["neck-rotation", "neck-extension"]
ModulusKind = Literal["G0", "E0"]
SimulationCase = Literal["A", "B"]

SIMULATION_CASE_ROTATION_AXES: dict[
    SimulationCase, tuple[float, float, float]
] = {
    "A": (0.0, 0.0, 1.0),
    "B": (1.0, 0.0, 0.0),
}

REFERENCE_FRAME_INTERVAL_SECONDS = 0.018
REFERENCE_FRAME_COUNT = 10
REFERENCE_DURATION_SECONDS = (
    (REFERENCE_FRAME_COUNT - 1) * REFERENCE_FRAME_INTERVAL_SECONDS
)
REFERENCE_MEAN_MAXIMUM_SHEAR_STRAIN: dict[ImpactMode, float] = {
    "neck-rotation": 0.017,
    "neck-extension": 0.011,
}
MIN_MAXWELL_BRANCHES = 3
MAX_MAXWELL_BRANCHES = 6
MIN_MODULUS_SCALE = 0.5
MAX_MODULUS_SCALE = 2.0
MIN_EQUILIBRIUM_RATIO = 0.001
MAX_EQUILIBRIUM_RATIO = 0.999
DEFAULT_TAU_MIN_SECONDS = 0.006
DEFAULT_TAU_MAX_SECONDS = 0.120


def logarithmic_relaxation_times(
    branch_count: int,
    *,
    minimum: float = DEFAULT_TAU_MIN_SECONDS,
    maximum: float = DEFAULT_TAU_MAX_SECONDS,
) -> tuple[float, ...]:
    """Return ``branch_count`` relaxation times spaced uniformly in log10."""
    if not MIN_MAXWELL_BRANCHES <= branch_count <= MAX_MAXWELL_BRANCHES:
        raise ValueError(
            f"branch_count must be in {MIN_MAXWELL_BRANCHES}.."
            f"{MAX_MAXWELL_BRANCHES}"
        )
    minimum = float(minimum)
    maximum = float(maximum)
    if (
        not np.isfinite((minimum, maximum)).all()
        or minimum <= 0.0
        or maximum <= minimum
    ):
        raise ValueError("tau limits must be finite, positive, and increasing")
    values = np.logspace(
        np.log10(minimum), np.log10(maximum), branch_count
    )
    values[0], values[-1] = minimum, maximum
    return tuple(float(value) for value in values)


def default_branch_fractions(
    branch_count: int,
    equilibrium_ratio: float,
) -> tuple[float, ...]:
    """Distribute ``1-r_inf`` over branches with decreasing positive weights."""
    if not MIN_MAXWELL_BRANCHES <= branch_count <= MAX_MAXWELL_BRANCHES:
        raise ValueError(
            f"branch_count must be in {MIN_MAXWELL_BRANCHES}.."
            f"{MAX_MAXWELL_BRANCHES}"
        )
    ratio = float(equilibrium_ratio)
    if not MIN_EQUILIBRIUM_RATIO <= ratio <= MAX_EQUILIBRIUM_RATIO:
        raise ValueError(
            f"equilibrium_ratio must be in {MIN_EQUILIBRIUM_RATIO}.."
            f"{MAX_EQUILIBRIUM_RATIO}"
        )
    weights = np.geomspace(1.0, 0.4, branch_count)
    fractions = (1.0 - ratio) * weights / np.sum(weights)
    return tuple(float(value) for value in fractions)


@dataclass(frozen=True, slots=True)
class GeneralizedMaxwellModel:
    r"""Prony parameterization of a scalar generalized-Maxwell solid.

    ``estimated_instantaneous_modulus`` is an estimated :math:`G_0` or
    :math:`E_0`; ``modulus_scale`` explores 0.5--2 times that estimate.
    ``equilibrium_ratio`` is :math:`r_\infty=G_\infty/G_0`, and each
    ``branch_fraction`` is :math:`g_i=G_i/G_0`. Consequently the required
    consistency condition is ``r_inf + sum(g_i) == 1``.
    """

    estimated_instantaneous_modulus: float = 2.0
    modulus_scale: float = 1.0
    equilibrium_ratio: float = 0.5
    branch_fractions: tuple[float, ...] = default_branch_fractions(3, 0.5)
    relaxation_times: tuple[float, ...] = logarithmic_relaxation_times(3)
    modulus_kind: ModulusKind = "G0"
    poisson_ratio: float = 0.49

    def __post_init__(self) -> None:
        estimated = float(self.estimated_instantaneous_modulus)
        scale = float(self.modulus_scale)
        ratio = float(self.equilibrium_ratio)
        branch_fractions = tuple(
            float(value) for value in self.branch_fractions
        )
        relaxation_times = tuple(
            float(value) for value in self.relaxation_times
        )
        poisson_ratio = float(self.poisson_ratio)
        object.__setattr__(self, "estimated_instantaneous_modulus", estimated)
        object.__setattr__(self, "modulus_scale", scale)
        object.__setattr__(self, "equilibrium_ratio", ratio)
        object.__setattr__(self, "branch_fractions", branch_fractions)
        object.__setattr__(self, "relaxation_times", relaxation_times)
        object.__setattr__(self, "poisson_ratio", poisson_ratio)

        if not np.isfinite(estimated) or estimated <= 0.0:
            raise ValueError(
                "estimated_instantaneous_modulus must be finite and positive"
            )
        if not MIN_MODULUS_SCALE <= scale <= MAX_MODULUS_SCALE:
            raise ValueError(
                f"modulus_scale must be in {MIN_MODULUS_SCALE}.."
                f"{MAX_MODULUS_SCALE}"
            )
        if not MIN_EQUILIBRIUM_RATIO <= ratio <= MAX_EQUILIBRIUM_RATIO:
            raise ValueError(
                f"equilibrium_ratio must be in {MIN_EQUILIBRIUM_RATIO}.."
                f"{MAX_EQUILIBRIUM_RATIO}"
            )
        if not (
            MIN_MAXWELL_BRANCHES
            <= len(branch_fractions)
            <= MAX_MAXWELL_BRANCHES
        ):
            raise ValueError(
                f"Maxwell branch count must be in {MIN_MAXWELL_BRANCHES}.."
                f"{MAX_MAXWELL_BRANCHES}"
            )
        if len(branch_fractions) != len(relaxation_times):
            raise ValueError(
                "branch_fractions and relaxation_times must have the same length"
            )
        if not np.isfinite(branch_fractions).all() or any(
            value <= 0.0 for value in branch_fractions
        ):
            raise ValueError("branch_fractions must be finite and positive")
        if not np.isclose(
            ratio + sum(branch_fractions), 1.0, rtol=1e-6, atol=1e-9
        ):
            raise ValueError(
                "equilibrium_ratio + sum(branch_fractions) must equal 1"
            )
        if not np.isfinite(relaxation_times).all() or any(
            value <= 0.0 for value in relaxation_times
        ):
            raise ValueError("relaxation_times must be finite and positive")
        if any(
            right <= left
            for left, right in zip(
                relaxation_times, relaxation_times[1:], strict=False
            )
        ):
            raise ValueError(
                "relaxation_times must be strictly increasing (log scale)"
            )
        if self.modulus_kind not in ("G0", "E0"):
            raise ValueError("modulus_kind must be 'G0' or 'E0'")
        if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be finite and in (-1, 0.5)")

    @property
    def branch_count(self) -> int:
        """Return the number of Maxwell branches, constrained to 3--6."""
        return len(self.branch_fractions)

    @property
    def estimated_shear_modulus(self) -> float:
        """Return the estimated instantaneous modulus converted to shear."""
        if self.modulus_kind == "G0":
            return self.estimated_instantaneous_modulus
        return self.estimated_instantaneous_modulus / (
            2.0 * (1.0 + self.poisson_ratio)
        )

    @property
    def instantaneous_modulus(self) -> float:
        """Return the active instantaneous shear modulus ``G0``."""
        return self.estimated_shear_modulus * self.modulus_scale

    @property
    def equilibrium_modulus(self) -> float:
        """Return ``G_inf = r_inf * G0``."""
        return self.equilibrium_ratio * self.instantaneous_modulus

    @property
    def branch_moduli(self) -> tuple[float, ...]:
        """Return ``G_i = g_i * G0`` for all Maxwell branches."""
        return tuple(
            fraction * self.instantaneous_modulus
            for fraction in self.branch_fractions
        )

    def relaxation_modulus(self, time: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate ``G(t)`` for non-negative times."""
        values = np.asarray(time, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("time must contain finite, non-negative values")
        moduli = np.asarray(self.branch_moduli, dtype=np.float64)
        relaxation = np.asarray(self.relaxation_times, dtype=np.float64)
        exponentials = np.exp(-values[..., np.newaxis] / relaxation)
        return np.asarray(
            self.equilibrium_modulus + exponentials @ moduli,
            dtype=np.float64,
        )


DEFAULT_MAXWELL_MODEL = GeneralizedMaxwellModel()


def generalized_maxwell_strain_response(
    times: npt.ArrayLike,
    shear_stress: npt.ArrayLike,
    *,
    model: GeneralizedMaxwellModel = DEFAULT_MAXWELL_MODEL,
) -> npt.NDArray[np.float64]:
    """Solve strain under a piecewise-linear prescribed shear-stress history.

    The first axis of ``shear_stress`` is time; any trailing dimensions are
    independent material points.  Maxwell-branch stresses are integrated with
    an exact exponential update for constant strain rate in each interval.
    """
    time_values = np.asarray(times, dtype=np.float64)
    stress_values = np.asarray(shear_stress, dtype=np.float64)
    if time_values.ndim != 1 or time_values.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if not np.isfinite(time_values).all():
        raise ValueError("times must contain only finite values")
    if time_values.size > 1 and np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if stress_values.ndim == 0 or stress_values.shape[0] != time_values.size:
        raise ValueError("shear_stress first axis must match times")
    if not np.isfinite(stress_values).all():
        raise ValueError("shear_stress must contain only finite values")

    strain = np.empty_like(stress_values, dtype=np.float64)
    branch_moduli = np.asarray(model.branch_moduli, dtype=np.float64)
    relaxation_times = np.asarray(model.relaxation_times, dtype=np.float64)
    branch_shape = (branch_moduli.size,) + (1,) * (stress_values.ndim - 1)
    shaped_moduli = branch_moduli.reshape(branch_shape)

    # A non-zero initial load is treated as an instantaneous stress step.
    strain[0] = stress_values[0] / model.instantaneous_modulus
    branch_stress = shaped_moduli * strain[0]

    for index in range(1, time_values.size):
        delta_time = float(time_values[index] - time_values[index - 1])
        decay = np.exp(-delta_time / relaxation_times)
        effective_branch_moduli = (
            branch_moduli
            * relaxation_times
            / delta_time
            * (1.0 - decay)
        )
        shaped_decay = decay.reshape(branch_shape)
        shaped_effective = effective_branch_moduli.reshape(branch_shape)
        previous_strain = strain[index - 1]

        numerator = stress_values[index] - np.sum(
            shaped_decay * branch_stress
            - shaped_effective * previous_strain,
            axis=0,
        )
        denominator = (
            model.equilibrium_modulus + float(np.sum(effective_branch_moduli))
        )
        current_strain = numerator / denominator
        branch_stress = (
            shaped_decay * branch_stress
            + shaped_effective * (current_strain - previous_strain)
        )
        strain[index] = current_strain

    return strain


def _simulation_times(
    *,
    frame_count: int,
    duration: float,
    times: npt.ArrayLike | None,
) -> npt.NDArray[np.float64]:
    if times is not None:
        values = np.asarray(times, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("times must be a non-empty one-dimensional array")
        if not np.isfinite(values).all():
            raise ValueError("times must contain only finite values")
        if values.size > 1 and np.any(np.diff(values) <= 0.0):
            raise ValueError("times must be strictly increasing")
        return values.copy()

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if duration < 0.0 or not np.isfinite(duration):
        raise ValueError("duration must be finite and non-negative")
    if frame_count > 1 and duration == 0.0:
        raise ValueError("duration must be positive when frame_count exceeds one")
    return np.linspace(0.0, duration, frame_count, dtype=np.float64)


def _rotational_spatial_weights(
    mesh: pv.DataSet,
    rotation_axis: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    if centers.shape != (mesh.n_cells, 3) or mesh.n_cells == 0:
        raise ValueError("mesh must contain spatial cells")
    if not np.isfinite(centers).all():
        raise ValueError("mesh cell centers must be finite")

    axis = np.asarray(rotation_axis, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("rotation_axis must contain three finite values")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        raise ValueError("rotation_axis cannot be zero")
    axis /= axis_norm

    centered = centers - np.mean(centers, axis=0)
    axial_component = np.outer(centered @ axis, axis)
    radial_distance = np.linalg.norm(centered - axial_component, axis=1)
    maximum = float(np.max(radial_distance))
    if maximum == 0.0 or not np.isfinite(maximum):
        raise ValueError("mesh cells have no extent perpendicular to rotation_axis")
    return np.asarray(radial_distance / maximum, dtype=np.float64)


def _impact_pulse(frame_count: int) -> npt.NDArray[np.float64]:
    """Return a unit signed pulse with loading and bounce-back lobes."""
    if frame_count == 1:
        return np.ones(1, dtype=np.float64)
    if frame_count == 2:
        return np.array((0.0, 1.0), dtype=np.float64)
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count, dtype=np.float64)
    return np.sin(phase)


def simulate_generalized_maxwell_strain(
    mesh: pv.DataSet,
    *,
    frame_count: int = REFERENCE_FRAME_COUNT,
    duration: float = REFERENCE_DURATION_SECONDS,
    times: npt.ArrayLike | None = None,
    model: GeneralizedMaxwellModel = DEFAULT_MAXWELL_MODEL,
    impact_mode: ImpactMode = "neck-rotation",
    target_mean_maximum_shear_strain: float | None = None,
    rotation_axis: npt.ArrayLike = (0.0, 0.0, 1.0),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate a reduced-order generalized-Maxwell maximum-shear field.

    A signed impact pulse is interpreted as prescribed shear stress.  Stress
    magnitude grows linearly with distance from ``rotation_axis`` as a simple
    rotational-inertia approximation. The load magnitude is calibrated once
    against the default Prony spectrum at the model's estimated modulus so its
    peak spatial mean matches the selected Gomez et al. atlas value. The load
    is then held fixed while the adjustable model is evaluated. Consequently,
    changing ``modulus_scale``, ``r_inf``, ``g_i``, or ``tau_i`` changes the
    simulated amplitude and/or time history instead of being normalized away.

    This is a material-point response on the displayed mesh, not an FE solve.
    """
    if impact_mode not in REFERENCE_MEAN_MAXIMUM_SHEAR_STRAIN:
        choices = ", ".join(REFERENCE_MEAN_MAXIMUM_SHEAR_STRAIN)
        raise ValueError(f"impact_mode must be one of: {choices}")
    target = (
        REFERENCE_MEAN_MAXIMUM_SHEAR_STRAIN[impact_mode]
        if target_mean_maximum_shear_strain is None
        else float(target_mean_maximum_shear_strain)
    )
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(
            "target_mean_maximum_shear_strain must be finite and positive"
        )

    time_values = _simulation_times(
        frame_count=frame_count,
        duration=float(duration),
        times=times,
    )
    spatial_weights = _rotational_spatial_weights(mesh, rotation_axis)
    unit_stress = (
        _impact_pulse(time_values.size)[:, np.newaxis]
        * spatial_weights[np.newaxis, :]
    )
    calibration_model = GeneralizedMaxwellModel(
        estimated_instantaneous_modulus=(
            model.estimated_instantaneous_modulus
        ),
        modulus_scale=1.0,
        equilibrium_ratio=DEFAULT_MAXWELL_MODEL.equilibrium_ratio,
        branch_fractions=DEFAULT_MAXWELL_MODEL.branch_fractions,
        relaxation_times=DEFAULT_MAXWELL_MODEL.relaxation_times,
        modulus_kind=model.modulus_kind,
        poisson_ratio=model.poisson_ratio,
    )
    calibration_strain = generalized_maxwell_strain_response(
        time_values,
        unit_stress,
        model=calibration_model,
    )
    calibration_peak = float(
        np.max(np.mean(np.abs(calibration_strain), axis=1))
    )
    if calibration_peak <= 0.0 or not np.isfinite(calibration_peak):
        raise ValueError("reference impact history produced no finite response")
    applied_stress = unit_stress * (target / calibration_peak)
    signed_strain = generalized_maxwell_strain_response(
        time_values,
        applied_stress,
        model=model,
    )
    maximum_shear_strain = np.abs(signed_strain)
    if not np.isfinite(maximum_shear_strain).all():
        raise ValueError("impact history produced no finite strain response")
    return time_values, np.asarray(maximum_shear_strain, dtype=np.float64)


def simulate_generalized_maxwell_cases(
    mesh: pv.DataSet,
    *,
    frame_count: int = REFERENCE_FRAME_COUNT,
    duration: float = REFERENCE_DURATION_SECONDS,
    times: npt.ArrayLike | None = None,
    model: GeneralizedMaxwellModel = DEFAULT_MAXWELL_MODEL,
    impact_mode: ImpactMode = "neck-rotation",
    target_mean_maximum_shear_strain: float | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    dict[SimulationCase, npt.NDArray[np.float64]],
]:
    """Generate the fixed A/B rotation-axis simulation presets.

    Both cases use the same material model, loading parameters, and time
    samples. Only the rotation axis changes: Case A uses ``(0, 0, 1)`` and
    Case B uses ``(1, 0, 0)``.
    """
    case_times, case_a = simulate_generalized_maxwell_strain(
        mesh,
        frame_count=frame_count,
        duration=duration,
        times=times,
        model=model,
        impact_mode=impact_mode,
        target_mean_maximum_shear_strain=(
            target_mean_maximum_shear_strain
        ),
        rotation_axis=SIMULATION_CASE_ROTATION_AXES["A"],
    )
    _, case_b = simulate_generalized_maxwell_strain(
        mesh,
        times=case_times,
        model=model,
        impact_mode=impact_mode,
        target_mean_maximum_shear_strain=(
            target_mean_maximum_shear_strain
        ),
        rotation_axis=SIMULATION_CASE_ROTATION_AXES["B"],
    )
    return case_times, {"A": case_a, "B": case_b}


__all__ = [
    "DEFAULT_MAXWELL_MODEL",
    "DEFAULT_TAU_MAX_SECONDS",
    "DEFAULT_TAU_MIN_SECONDS",
    "GeneralizedMaxwellModel",
    "ImpactMode",
    "MAX_MAXWELL_BRANCHES",
    "MAX_MODULUS_SCALE",
    "MAX_EQUILIBRIUM_RATIO",
    "MIN_MAXWELL_BRANCHES",
    "MIN_MODULUS_SCALE",
    "MIN_EQUILIBRIUM_RATIO",
    "ModulusKind",
    "REFERENCE_DURATION_SECONDS",
    "REFERENCE_FRAME_COUNT",
    "REFERENCE_FRAME_INTERVAL_SECONDS",
    "REFERENCE_MEAN_MAXIMUM_SHEAR_STRAIN",
    "SIMULATION_CASE_ROTATION_AXES",
    "SimulationCase",
    "default_branch_fractions",
    "generalized_maxwell_strain_response",
    "logarithmic_relaxation_times",
    "simulate_generalized_maxwell_cases",
    "simulate_generalized_maxwell_strain",
]
