"""Tests for the reduced-order generalized-Maxwell simulation."""

import unittest

import numpy as np
import pyvista as pv

from brain_strain.simulation import (
    REFERENCE_DURATION_SECONDS,
    REFERENCE_FRAME_COUNT,
    GeneralizedMaxwellModel,
    default_branch_fractions,
    generalized_maxwell_strain_response,
    logarithmic_relaxation_times,
    simulate_generalized_maxwell_strain,
)


class GeneralizedMaxwellTests(unittest.TestCase):
    def test_relaxation_modulus_has_correct_limits(self) -> None:
        model = GeneralizedMaxwellModel(
            estimated_instantaneous_modulus=10.0,
            equilibrium_ratio=0.2,
            branch_fractions=(0.3, 0.3, 0.2),
            relaxation_times=(0.1, 1.0, 10.0),
        )

        values = model.relaxation_modulus((0.0, 1000.0))

        self.assertAlmostEqual(values[0], 10.0)
        self.assertAlmostEqual(values[1], 2.0)

    def test_constant_stress_creeps_from_instantaneous_to_equilibrium_response(
        self,
    ) -> None:
        model = GeneralizedMaxwellModel(
            estimated_instantaneous_modulus=5.0,
            equilibrium_ratio=0.4,
            branch_fractions=(0.2, 0.2, 0.2),
            relaxation_times=(0.05, 0.1, 0.2),
        )
        times = np.linspace(0.0, 5.0, 5001)
        stress = np.ones(times.size)

        strain = generalized_maxwell_strain_response(
            times, stress, model=model
        )

        self.assertAlmostEqual(strain[0], 1.0 / 5.0)
        self.assertAlmostEqual(strain[-1], 1.0 / 2.0, places=3)
        self.assertTrue(np.all(np.diff(strain) >= -1e-12))

    def test_model_rejects_mismatched_or_nonpositive_branches(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            GeneralizedMaxwellModel(
                branch_fractions=(0.2, 0.2, 0.1),
                relaxation_times=(0.01, 0.1, 1.0, 10.0),
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            GeneralizedMaxwellModel(
                branch_fractions=(0.25, 0.25, 0.0),
                relaxation_times=(0.01, 0.1, 1.0),
            )

    def test_mesh_response_uses_reference_sampling_and_amplitude(self) -> None:
        mesh = pv.ImageData(dimensions=(5, 5, 5))

        times, strain = simulate_generalized_maxwell_strain(mesh)

        self.assertEqual(times.shape, (REFERENCE_FRAME_COUNT,))
        self.assertAlmostEqual(times[-1], REFERENCE_DURATION_SECONDS)
        self.assertEqual(strain.shape, (REFERENCE_FRAME_COUNT, mesh.n_cells))
        self.assertTrue(np.isfinite(strain).all())
        self.assertTrue(np.all(strain >= 0.0))
        self.assertAlmostEqual(float(np.max(np.mean(strain, axis=1))), 0.017)
        np.testing.assert_allclose(strain[0], 0.0)

    def test_neck_extension_uses_its_reported_reference_scale(self) -> None:
        mesh = pv.ImageData(dimensions=(4, 4, 4))

        _, strain = simulate_generalized_maxwell_strain(
            mesh, impact_mode="neck-extension"
        )

        self.assertAlmostEqual(float(np.max(np.mean(strain, axis=1))), 0.011)

    def test_explicit_times_are_preserved(self) -> None:
        mesh = pv.ImageData(dimensions=(4, 4, 4))
        requested = np.array((0.0, 0.01, 0.025, 0.08, 0.2))

        times, strain = simulate_generalized_maxwell_strain(
            mesh, times=requested
        )

        np.testing.assert_array_equal(times, requested)
        self.assertEqual(strain.shape, (requested.size, mesh.n_cells))

    def test_relaxation_times_change_the_transient_shape(self) -> None:
        mesh = pv.ImageData(dimensions=(4, 4, 4))
        fast = GeneralizedMaxwellModel(
            branch_fractions=(0.25, 0.15, 0.10),
            relaxation_times=(0.0001, 0.0003, 0.001),
        )
        slow = GeneralizedMaxwellModel(
            branch_fractions=(0.25, 0.15, 0.10),
            relaxation_times=(1.0, 3.0, 10.0),
        )

        _, fast_strain = simulate_generalized_maxwell_strain(mesh, model=fast)
        _, slow_strain = simulate_generalized_maxwell_strain(mesh, model=slow)

        self.assertFalse(np.allclose(fast_strain, slow_strain))

    def test_branch_count_and_prony_sum_are_constrained(self) -> None:
        for count in range(3, 7):
            model = GeneralizedMaxwellModel(
                branch_fractions=default_branch_fractions(count, 0.4),
                equilibrium_ratio=0.4,
                relaxation_times=logarithmic_relaxation_times(count),
            )
            self.assertEqual(model.branch_count, count)

        with self.assertRaisesRegex(ValueError, "branch count"):
            GeneralizedMaxwellModel(
                branch_fractions=(0.25, 0.25),
                relaxation_times=(0.01, 0.1),
            )
        with self.assertRaisesRegex(ValueError, "must equal 1"):
            GeneralizedMaxwellModel(
                branch_fractions=(0.1, 0.1, 0.1),
                relaxation_times=(0.01, 0.1, 1.0),
            )

    def test_relaxation_times_are_logarithmically_spaced(self) -> None:
        values = np.asarray(
            logarithmic_relaxation_times(6, minimum=1e-4, maximum=10.0)
        )

        np.testing.assert_allclose(
            np.diff(np.log10(values)),
            np.full(5, 1.0),
        )

    def test_modulus_scale_changes_response_amplitude(self) -> None:
        mesh = pv.ImageData(dimensions=(4, 4, 4))
        soft = GeneralizedMaxwellModel(modulus_scale=0.5)
        stiff = GeneralizedMaxwellModel(modulus_scale=2.0)

        _, soft_strain = simulate_generalized_maxwell_strain(mesh, model=soft)
        _, stiff_strain = simulate_generalized_maxwell_strain(
            mesh, model=stiff
        )
        soft_peak = float(np.max(np.mean(soft_strain, axis=1)))
        stiff_peak = float(np.max(np.mean(stiff_strain, axis=1)))

        self.assertAlmostEqual(soft_peak / stiff_peak, 4.0)

    def test_modulus_and_equilibrium_ratio_ranges_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "modulus_scale"):
            GeneralizedMaxwellModel(modulus_scale=0.49)
        with self.assertRaisesRegex(ValueError, "modulus_scale"):
            GeneralizedMaxwellModel(modulus_scale=2.01)
        with self.assertRaisesRegex(ValueError, "equilibrium_ratio"):
            GeneralizedMaxwellModel(equilibrium_ratio=0.0)

    def test_youngs_modulus_is_converted_to_shear_modulus(self) -> None:
        model = GeneralizedMaxwellModel(
            estimated_instantaneous_modulus=2980.0,
            modulus_kind="E0",
            poisson_ratio=0.49,
        )

        self.assertAlmostEqual(model.instantaneous_modulus, 1000.0)


if __name__ == "__main__":
    unittest.main()
