import copy
import unittest
from mast.data import validate_records, grouped_split
from mast.train_baseline import evaluate


def records():
    return [dict(record_id=f'r{i}', group_id=f'g{i//2}', family='Low-alloy steel',
                 source=f'fixture-{i}', source_type='experimental', license='test-only', test_method='fixture',
                 features={'temperature': {'value': 400+i, 'unit': 'degC'}},
                 properties={'UTS': {'value': 1000-i*2, 'unit': 'MPa'}}) for i in range(20)]


class DataTests(unittest.TestCase):
    def test_groups_cannot_leak(self):
        split = grouped_split(records())
        groups = {k: {r['group_id'] for r in v} for k, v in split.items()}
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['test'] & groups['validation'])
        self.assertEqual(sum(map(len, split.values())), 20)

    def test_missing_provenance_and_duplicate_observation(self):
        r = records()
        r[0].pop('source')
        with self.assertRaises(ValueError): validate_records(r)
        r = records()
        twin = copy.deepcopy(r[0]); twin['record_id'] = 'new'; twin['group_id'] = 'new'
        with self.assertRaises(ValueError): validate_records(r + [twin])

    def test_measured_benchmark_rejects_synthetic(self):
        r = records(); r[0]['source_type'] = 'synthetic'
        with self.assertRaises(ValueError): evaluate(r)

    def test_units_are_not_silently_mixed(self):
        r = records(); r[0]['properties']['UTS']['unit'] = 'psi'
        with self.assertRaises(ValueError): evaluate(r)

    def test_baseline_reports_per_property_units(self):
        report = evaluate(records())
        self.assertEqual(report['metrics']['test']['UTS']['unit'], 'MPa')
        self.assertIn('training_mean_mae', report['metrics']['test']['UTS'])


if __name__ == '__main__': unittest.main()
