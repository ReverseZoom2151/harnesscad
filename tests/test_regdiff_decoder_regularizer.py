"""Tests for numeric.regdiff_decoder_regularizer (CADiffusion Sec. 3.3, Eq. 8)."""

import unittest

from numeric.diffusioncad_sqrt_schedule import SqrtNoiseSchedule
from numeric.regdiff_decoder_regularizer import (
    batch_regularization_loss,
    combined_decoder_loss,
    decoder_distance,
    regularization_energy,
)


def _const_eps(const):
    def eps_model(z, t):
        return const
    return eps_model


def _identity_decoder(z):
    return list(z)


class TestDecoderDistance(unittest.TestCase):
    def test_l2_distance(self):
        self.assertAlmostEqual(decoder_distance([3.0, 4.0], [0.0, 0.0]), 5.0)

    def test_zero_on_equal(self):
        self.assertEqual(decoder_distance([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            decoder_distance([1.0], [1.0, 2.0])


class TestRegularizationEnergy(unittest.TestCase):
    def setUp(self):
        self.sched = SqrtNoiseSchedule(steps=40)
        self.eps = _const_eps([0.03, -0.02, 0.05, 0.01])
        self.z0 = [0.4, -0.1, 0.2, 0.6]

    def test_sigma_zero_recovers_clean_decode(self):
        # With sigma=0 the invert/regenerate round trip (constant eps) is exact,
        # so the identity decoder reproduces z0 and the energy equals ||z0 - cad||.
        cad = list(self.z0)
        energy = regularization_energy(
            self.z0, cad, self.sched, self.eps, _identity_decoder,
            total_steps=40, sigma=0.0, seed=1,
        )
        self.assertAlmostEqual(energy, 0.0, places=6)

    def test_perturbation_increases_energy(self):
        cad = list(self.z0)
        e0 = regularization_energy(
            self.z0, cad, self.sched, self.eps, _identity_decoder,
            total_steps=40, sigma=0.0, seed=1,
        )
        e1 = regularization_energy(
            self.z0, cad, self.sched, self.eps, _identity_decoder,
            total_steps=40, sigma=0.2, seed=1,
        )
        self.assertGreater(e1, e0)

    def test_deterministic_given_seed(self):
        cad = [0.0, 0.0, 0.0, 0.0]
        a = regularization_energy(
            self.z0, cad, self.sched, self.eps, _identity_decoder,
            40, 0.15, seed=7,
        )
        b = regularization_energy(
            self.z0, cad, self.sched, self.eps, _identity_decoder,
            40, 0.15, seed=7,
        )
        self.assertEqual(a, b)

    def test_different_seed_changes_energy(self):
        cad = [0.0, 0.0, 0.0, 0.0]
        a = regularization_energy(
            self.z0, cad, self.sched, self.eps, _identity_decoder,
            40, 0.2, seed=7,
        )
        b = regularization_energy(
            self.z0, cad, self.sched, self.eps, _identity_decoder,
            40, 0.2, seed=99,
        )
        self.assertNotEqual(a, b)


class TestBatchAndCombined(unittest.TestCase):
    def setUp(self):
        self.sched = SqrtNoiseSchedule(steps=30)
        self.eps = _const_eps([0.01, 0.02])

    def test_batch_average(self):
        latents = [[0.1, 0.2], [0.3, -0.4]]
        cads = [[0.1, 0.2], [0.3, -0.4]]
        loss = batch_regularization_loss(
            latents, cads, self.sched, self.eps, _identity_decoder,
            30, sigma=0.0, seed=0,
        )
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_batch_empty(self):
        loss = batch_regularization_loss(
            [], [], self.sched, self.eps, _identity_decoder, 30, 0.1, 0,
        )
        self.assertEqual(loss, 0.0)

    def test_batch_length_mismatch(self):
        with self.assertRaises(ValueError):
            batch_regularization_loss(
                [[0.1, 0.2]], [], self.sched, self.eps, _identity_decoder,
                30, 0.1, 0,
            )

    def test_combined_reg_weight_zero_is_recon_only(self):
        z0 = [0.5, -0.5]
        cad = [0.0, 0.0]
        loss = combined_decoder_loss(
            z0, cad, self.sched, self.eps, _identity_decoder,
            30, sigma=0.2, seed=1, reg_weight=0.0,
        )
        # recon only = ||z0 - cad|| = sqrt(0.5)
        self.assertAlmostEqual(loss, decoder_distance(z0, cad))

    def test_combined_adds_weighted_regularizer(self):
        z0 = [0.5, -0.5]
        cad = [0.0, 0.0]
        recon = decoder_distance(z0, cad)
        full = combined_decoder_loss(
            z0, cad, self.sched, self.eps, _identity_decoder,
            30, sigma=0.2, seed=1, reg_weight=1.0,
        )
        self.assertGreater(full, recon)

    def test_combined_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            combined_decoder_loss(
                [0.1], [0.0], self.sched, self.eps, _identity_decoder,
                30, 0.1, 1, reg_weight=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
