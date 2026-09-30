import unittest
import numpy as np
from mast.catalog import STEEL_PRESETS, PROPERTY_NAMES, PROP_RANGES
from mast.engine import Constraints, validate_composition, validate_targets, candidate_mask, pareto_mask, load_models, recommend
from mast.physics import martensite_fraction
from pathlib import Path
import torch


def composition(name='AISI 4340'):
    c = dict(STEEL_PRESETS[name])
    c['Fe'] = 100-sum(c.values())
    return c


class ValidationTests(unittest.TestCase):
    def test_h13_is_rejected_without_clipping(self):
        with self.assertRaisesRegex(ValueError, 'Si.*Cr.*Mo.*V'):
            validate_composition(composition('H13 Tool'))

    def test_composition_balance_and_nonfinite(self):
        c = composition()
        validate_composition(c)
        c['Fe'] += 1
        with self.assertRaises(ValueError): validate_composition(c)
        c['Fe'] = float('nan')
        with self.assertRaises(ValueError): validate_composition(c)

    def test_impossible_target_order(self):
        t = {p: PROP_RANGES[p][2] for p in PROPERTY_NAMES}
        t['YS'] = t['UTS'] + 10
        with self.assertRaises(ValueError): validate_targets(t)

    def test_process_and_output_screening(self):
        p = np.array([[850, 1, 1, 450, 2]]*6, dtype=float)
        y = np.array([[1300, 1100, 10, 15, 40, 80]]*6, dtype=float)
        v = np.ones((6, 6))
        p[1, 0] = 780  # below Ac3 + margin
        p[2, 2] = 1.5  # not a real quench category
        y[3, 1] = 2000  # YS > UTS
        v[4, 0] = -1
        p[5, 4] = 8  # exceeds user combined hold limit
        np.testing.assert_array_equal(candidate_mask(p, y, v, 780, Constraints(max_hold_hours=4)), [True, False, False, False, False, False])

    def test_geometry_and_equipment(self):
        for constraints in (Constraints(section_diameter_mm=50), Constraints(allowed_quenches=()), Constraints(max_hold_hours=float('nan'))):
            with self.assertRaises(ValueError): constraints.validate()

    def test_pareto_tradeoffs(self):
        np.testing.assert_array_equal(pareto_mask([[1, 3], [2, 2], [3, 3], [1, 3]]), [True, True, False, True])

    def test_martensite_conserves_available_phase(self):
        fractions = martensite_fraction(350, [400, 300, 25], .7)
        self.assertEqual(fractions[0], 0)
        self.assertTrue(np.all(np.diff(fractions) >= 0))
        self.assertTrue(np.all(fractions <= .7))
        self.assertEqual(float(martensite_fraction(350, 25, 0)), 0)
        with self.assertRaises(ValueError): martensite_fraction(350, 25, 1.1)


class CheckpointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        cls.models = load_models(root/'checkpoints/forward_model.pt', root/'checkpoints/policy.pt')

    def test_real_checkpoints_reproducible_and_traceable(self):
        c = composition()
        t = {p: PROP_RANGES[p][2] for p in PROPERTY_NAMES}
        before = torch.random.get_rng_state().clone()
        a = recommend(self.models, c, t, n_candidates=20, top_k=3)
        b = recommend(self.models, c, t, n_candidates=20, top_k=3)
        self.assertEqual(a['results'], b['results'])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(a['results'])
        self.assertEqual(len(a['model_hashes']['forward_sha256']), 64)
        self.assertEqual(a['evidence_status'], 'synthetic_research_only')

    def test_no_feasible_process_never_returns_recipe(self):
        t = {p: PROP_RANGES[p][2] for p in PROPERTY_NAMES}
        report = recommend(self.models, composition(), t, n_candidates=20,
                           method='policy', constraints=Constraints(max_temperature_c=1))
        self.assertEqual(report['results'], [])
        self.assertEqual(report['valid_candidates'], 0)


if __name__ == '__main__': unittest.main()
