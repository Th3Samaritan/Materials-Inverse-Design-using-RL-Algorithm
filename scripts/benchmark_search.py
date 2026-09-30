"""Equal-budget surrogate comparison. Run from repository root.

No experimental truth is available in this benchmark.
"""
import json
from pathlib import Path
import sys
from time import perf_counter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mast.catalog import STEEL_PRESETS, PROPERTY_NAMES, PROP_RANGES
from mast.engine import load_models, recommend


def main():
    torch.set_num_threads(2)
    models = load_models(ROOT/'checkpoints/forward_model.pt', ROOT/'checkpoints/policy.pt')
    rows = []
    for grade in ('AISI 4340', 'AISI 1045', 'AISI 4140'):
        comp = dict(STEEL_PRESETS[grade]); comp['Fe'] = 100-sum(comp.values())
        for seed in (7, 42, 123):
            for method in ('policy', 'random', 'hybrid'):
                start = perf_counter()
                report = recommend(models, comp, {p: PROP_RANGES[p][2] for p in PROPERTY_NAMES},
                                   n_candidates=300, seed=seed, method=method)
                best = report['results'][0] if report['results'] else None
                rows.append(dict(grade=grade, seed=seed, method=method, seconds=perf_counter()-start,
                                 valid_candidates=report['valid_candidates'],
                                 best_score=best['score'] if best else None,
                                 any_displayed_within_tolerance=any(r['within_tolerance'] for r in report['results'])))
    output = dict(evidence='surrogate_only_not_experimental_validation', candidate_budget=300,
                  cpu_threads=2, torch_version=torch.__version__, model_hashes=models[-1], results=rows)
    destination = ROOT/'outputs/search_benchmark.json'
    destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
    for method in ('policy', 'random', 'hybrid'):
        subset = [r for r in rows if r['method'] == method]
        print(method, 'mean best score', sum(r['best_score'] for r in subset)/len(subset),
              'mean seconds', sum(r['seconds'] for r in subset)/len(subset),
              'tolerance matches', sum(r['any_displayed_within_tolerance'] for r in subset), '/', len(subset))


if __name__ == '__main__': main()
