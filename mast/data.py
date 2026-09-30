"""Explicit provenance and group splits for future family-specific datasets."""
import math
import random
from .engine import FAMILIES


def validate_records(records):
    """Validate normalized JSON records; preserve units instead of guessing conversions."""
    if not isinstance(records, list) or not records:
        raise ValueError('Dataset is empty.')
    ids = set()
    signatures = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Each record must be an object.')
        required = ('record_id', 'group_id', 'family', 'source', 'source_type', 'license', 'features', 'properties', 'test_method')
        if any(k not in row or not row[k] for k in required):
            raise ValueError('Each record needs identifiers, family, source, source_type, license, test_method, features and properties.')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in required if k not in ('features', 'properties')):
            raise ValueError('Identifiers and provenance fields must be nonempty strings.')
        if row['record_id'] in ids:
            raise ValueError('Duplicate record_id: ' + row['record_id'])
        ids.add(row['record_id'])
        if row['family'] not in FAMILIES or row['source_type'] not in ('experimental', 'simulation', 'synthetic'):
            raise ValueError('Unknown family or evidence type.')
        for section in ('features', 'properties'):
            if not isinstance(row[section], dict):
                raise ValueError('Features and properties must be named measurement mappings.')
            for name, measurement in row[section].items():
                if not isinstance(measurement, dict) or set(measurement) != {'value', 'unit'}:
                    raise ValueError(f'{name} needs value and unit.')
                value = measurement['value']
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not isinstance(measurement['unit'], str) or not measurement['unit'].strip()):
                    raise ValueError(f'{name} needs a finite numeric value and an explicit unit.')
        # Exact duplicate observations should not leak across groups.
        signature = (row['source'], tuple((section, name, m['value'], m['unit'])
                     for section in ('features', 'properties') for name, m in sorted(row[section].items())))
        if signature in signatures:
            raise ValueError('Duplicate observation; retain genuine replicates with distinct source identifiers.')
        signatures.add(signature)
    return records


def grouped_split(records, seed=42):
    """Split by heat/lot/campaign, never random rows. Minimum five groups."""
    validate_records(records)
    groups = sorted({r['group_id'] for r in records})
    if len(groups) < 5:
        raise ValueError('At least five independent groups are required for this research split.')
    random.Random(seed).shuffle(groups)
    n = max(1, len(groups)//5)
    selected = {'test': set(groups[:n]), 'validation': set(groups[n:2*n]), 'train': set(groups[2*n:])}
    return {name: [r for r in records if r['group_id'] in group] for name, group in selected.items()}
