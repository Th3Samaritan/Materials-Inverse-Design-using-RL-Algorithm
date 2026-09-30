"""Train and evaluate a measured-data baseline; never promote it automatically.

Usage: python -m mast.train_baseline data.json --output outputs/baseline
Output is a report, not an executable pickle artifact or a deployed predictor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from .data import grouped_split


def evaluate(records, seed=42):
    splits = grouped_split(records, seed)
    if any(r['source_type'] != 'experimental' for r in records):
        raise ValueError('This measured-data benchmark rejects synthetic and simulated observations.')
    if len({r['family'] for r in records}) != 1:
        raise ValueError('Train a separate baseline per material family.')
    features = sorted(records[0]['features'])
    properties = sorted(records[0]['properties'])
    for row in records:
        for section, names in (('features', features), ('properties', properties)):
            if sorted(row[section]) != names or any(row[section][k]['unit'] != records[0][section][k]['unit'] for k in names):
                raise ValueError('Feature/property names and units must match across this dataset.')
        if row['test_method'] != records[0]['test_method']:
            raise ValueError('Stratify by test method before this baseline evaluation.')
    def arrays(rows):
        return (np.array([[r['features'][k]['value'] for k in features] for r in rows]),
                np.array([[r['properties'][k]['value'] for k in properties] for r in rows]))
    x, y = arrays(splits['train'])
    # A fixed baseline: validation data does not tune hyperparameters implicitly.
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)
    model.fit(x, y.ravel() if len(properties) == 1 else y)
    output = dict(status='research_baseline_only', seed=seed, family=records[0]['family'],
                  features=features, properties=properties, test_method=records[0]['test_method'],
                  split_groups={k: sorted({r['group_id'] for r in rows}) for k, rows in splits.items()}, metrics={})
    for name in ('validation', 'test'):
        xx, yy = arrays(splits[name])
        prediction = model.predict(xx).reshape(len(xx), len(properties))
        mean_prediction = np.broadcast_to(y.mean(0), yy.shape)
        output['metrics'][name] = {prop: dict(
            unit=records[0]['properties'][prop]['unit'],
            random_forest_mae=float(mean_absolute_error(yy[:, i], prediction[:, i])),
            training_mean_mae=float(mean_absolute_error(yy[:, i], mean_prediction[:, i])))
            for i, prop in enumerate(properties)}
    output['limitations'] = ['No calibrated intervals; no industrial qualification.',
                            'Group IDs are supplied by the data owner and need independent audit.',
                            'Five groups is a software minimum, not evidence of statistical sufficiency.']
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/baseline'))
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    report = evaluate(json.loads(raw))
    report['dataset_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(args.output/'report.json')


if __name__ == '__main__':
    main()
