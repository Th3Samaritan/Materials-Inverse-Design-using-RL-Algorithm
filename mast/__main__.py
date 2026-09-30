"""Headless reproducible legacy checkpoint demonstration."""
import argparse
import json
from pathlib import Path
from .catalog import STEEL_PRESETS, PROPERTY_NAMES, PROP_RANGES
from .engine import load_models, recommend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['policy', 'random', 'hybrid'], default='hybrid')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--candidates', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('outputs/research_run.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    models = load_models(root/'checkpoints/forward_model.pt', root/'checkpoints/policy.pt')
    comp = dict(STEEL_PRESETS['AISI 4340'])
    comp['Fe'] = 100-sum(comp.values())
    report = recommend(models, comp, {p: PROP_RANGES[p][2] for p in PROPERTY_NAMES},
                       seed=args.seed, n_candidates=args.candidates, method=args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"{args.output}: {report['valid_candidates']} screened candidates; research only")


if __name__ == '__main__':
    main()
