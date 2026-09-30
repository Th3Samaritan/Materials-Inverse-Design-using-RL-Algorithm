"""Validated, reproducible research inference. No manufacturing certification."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import io
import math
import threading
from pathlib import Path
import numpy as np
import torch
from .catalog import ELEMENT_NAMES, PROPERTY_NAMES, COMP_RANGES, QUENCH_MAP
from .models import ForwardEnsemble, AutoregressivePolicy, compute_ac3_torch

_SAMPLING_LOCK = threading.RLock()


FAMILIES = {
    'Low-alloy steel': ('research', 'Legacy synthetic quench-and-temper model; experimental validation required.'),
    'Tool and stainless steels': ('planned', 'Needs grade-specific phase, carbide and transformation models and measured data.'),
    'Fiber composites': ('planned', 'Needs constituent, layup, fiber fraction, void content and cure-history data.'),
    'Thermoplastics': ('planned', 'Needs chemistry, molecular weight, crystallinity and processing-history data.'),
    'Elastomers': ('planned', 'Needs formulation in phr, crosslinking, cure, strain-rate and aging data.'),
}


@dataclass(frozen=True)
class Constraints:
    max_temperature_c: float = 1100.0
    max_hold_hours: float = 12.0
    allowed_quenches: tuple = (0, 1, 2, 3)
    section_diameter_mm: float = 25.0

    def validate(self):
        if not all(math.isfinite(x) for x in (self.max_temperature_c, self.max_hold_hours, self.section_diameter_mm)):
            raise ValueError('Equipment constraints must be finite.')
        if self.max_temperature_c <= 0 or self.max_hold_hours <= 0:
            raise ValueError('Temperature and hold-time limits must be positive.')
        if self.section_diameter_mm != 25.0:
            raise ValueError('Legacy model has no geometry input; only its nominal 25 mm synthetic scenario is supported.')
        if not self.allowed_quenches or any(type(q) is not int or q not in QUENCH_MAP for q in self.allowed_quenches):
            raise ValueError('Select at least one supported quench medium.')


def validate_composition(comp):
    if set(comp) != set(ELEMENT_NAMES):
        raise ValueError('Composition must contain exactly the 15 legacy elements, including Fe balance.')
    if any(not math.isfinite(v) or v < 0 for v in comp.values()):
        raise ValueError('Composition values must be finite, nonnegative weight percentages.')
    if abs(sum(comp.values()) - 100) > 1e-3:
        raise ValueError('Composition must sum to 100 wt%.')
    outside = [e for e, (lo, hi) in COMP_RANGES.items() if not lo <= comp[e] <= hi]
    if not 90 <= comp['Fe'] <= 99.9:
        outside.append('Fe')
    if outside:
        raise ValueError('Outside legacy synthetic training bounds: ' + ', '.join(outside))


def validate_targets(targets):
    if set(targets) != set(PROPERTY_NAMES) or any(not math.isfinite(v) or v <= 0 for v in targets.values()):
        raise ValueError('All six targets must be finite and positive.')
    if targets['YS'] > targets['UTS']:
        raise ValueError('Yield strength cannot exceed UTS for this tensile-property definition.')
    if targets['elongation'] > 100 or targets['reduction_of_area'] > 100 or targets['HRC'] > 70:
        raise ValueError('Targets exceed the supported property domain.')


def load_models(forward_source, policy_source):
    """Use restricted tensor loading, never arbitrary checkpoint pickle execution."""
    payloads = [s if isinstance(s, bytes) else Path(s).read_bytes() for s in (forward_source, policy_source)]
    f, p = [torch.load(io.BytesIO(b), map_location='cpu', weights_only=True) for b in payloads]
    forward, actor = ForwardEnsemble(), AutoregressivePolicy()
    forward.load_state_dict(f['model_state_dict'], strict=True)
    actor.load_state_dict(p['actor_state_dict'], strict=True)
    for model in (forward, actor):
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError('Checkpoint contains non-finite weights or normalization.')
        model.eval()
    for name in ('input_std', 'output_std'):
        if (getattr(forward, name) <= 0).any():
            raise ValueError('Checkpoint normalization scales must be positive.')
    mean, std = p.get('prop_mean'), p.get('prop_std')
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise ValueError('Policy checkpoint requires property normalization metadata.')
    if mean.shape != (6,) or std.shape != (6,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid policy normalization metadata.')
    hashes = dict(zip(('forward_sha256', 'policy_sha256'), (hashlib.sha256(b).hexdigest() for b in payloads)))
    return forward, actor, mean, std, torch.device('cpu'), hashes


def candidate_mask(processes, means, variances, ac3, constraints):
    constraints.validate()
    p, y, v = map(np.asarray, (processes, means, variances))
    good = np.isfinite(p).all(1) & np.isfinite(y).all(1) & np.isfinite(v).all(1) & (v >= 0).all(1)
    # Strict Ac3 offsets can reject samples produced by the legacy temperature clamp.
    good &= (p[:, 0] >= max(750, ac3 + 30)) & (p[:, 0] <= min(1100, ac3 + 150, constraints.max_temperature_c))
    good &= (p[:, 1] >= .5) & (p[:, 1] <= 4) & (p[:, 4] >= .5) & (p[:, 4] <= 8)
    good &= (p[:, 3] >= 150) & (p[:, 3] <= np.minimum(700, p[:, 0] - 20))
    good &= (p[:, 1] + p[:, 4] <= constraints.max_hold_hours) & np.isin(p[:, 2], constraints.allowed_quenches)
    good &= (y >= 0).all(1) & (y[:, 1] <= y[:, 0]) & (y[:, 2:4] <= 100).all(1) & (y[:, 4] <= 70)
    return good


def pareto_mask(objectives):
    """Minimization; preserve identical nondominated points."""
    a = np.asarray(objectives)
    return np.array([not np.any(np.all(a <= row, axis=1) & np.any(a < row, axis=1)) for row in a])


def recommend(models, comp, targets, *, n_candidates=300, top_k=4, seed=42,
              constraints=None, method='hybrid', tolerance_fraction=.15, time_weight=0.0):
    validate_composition(comp)
    validate_targets(targets)
    constraints = constraints or Constraints()
    constraints.validate()
    if method not in ('policy', 'random', 'hybrid'):
        raise ValueError('Unknown search method.')
    if (type(top_k) is not int or type(n_candidates) is not int
        or not 1 <= top_k <= n_candidates <= 10000 or not 0 < tolerance_fraction <= 1
        or not math.isfinite(time_weight) or time_weight < 0
        or type(seed) is not int or not 0 <= seed <= 2147483647):
        raise ValueError('Invalid candidate budget, tolerance or time weight.')
    fwd, actor, pm, ps, device, hashes = models
    c = torch.tensor([comp[e] for e in ELEMENT_NAMES], dtype=torch.float32).expand(n_candidates, -1)
    t = torch.tensor([targets[p] for p in PROPERTY_NAMES], dtype=torch.float32).expand(n_candidates, -1)
    ac3 = compute_ac3_torch(c)
    # fork_rng restores the caller's RNG state after reproducible research sampling.
    with _SAMPLING_LOCK, torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(seed)
        count_policy = n_candidates if method == 'policy' else (n_candidates // 2 if method == 'hybrid' else 0)
        parts = []
        if count_policy:
            parts.append(actor(c[:count_policy], t[:count_policy], ac3[:count_policy], temperature=.6)[0])
        count_random = n_candidates - count_policy
        if count_random:
            lo, hi = max(750, ac3[0].item()+30), min(1100, ac3[0].item()+150, constraints.max_temperature_c)
            if hi < lo:
                raise ValueError('No austenitizing window within equipment and model constraints.')
            r = torch.rand(count_random, 5)
            r[:, 0] = lo + r[:, 0] * (hi-lo)
            r[:, 1] = .5 + r[:, 1]*3.5
            allowed = torch.tensor(constraints.allowed_quenches)
            r[:, 2] = allowed[torch.randint(len(allowed), (count_random,))]
            r[:, 3] = 150 + r[:, 3]*550
            r[:, 4] = .5 + r[:, 4]*7.5
            parts.append(r)
        processes = torch.cat(parts)
        means, variances = fwd(c, processes)
    p, y, v = processes.numpy(), means.numpy(), variances.numpy()
    valid = candidate_mask(p, y, v, ac3[0].item(), constraints)
    ids = np.flatnonzero(valid)
    warnings = ['Research only: synthetic legacy model, no demonstrated industrial qualification.',
                'Ensemble spread is not a calibrated prediction interval.',
                'Hold time excludes heating ramps, transfer, quenching and cooling; it is not total cycle time.',
                'Training-source phase-fraction defect identified; checkpoint lineage is not fully verified.']
    results = []
    if len(ids):
        scale = ps.numpy()
        target = t[0].numpy()
        loss = (((y[ids]-target)/scale)**2).mean(1)
        spread = (v[ids]/scale**2).mean(1)
        hours = p[ids, 1]+p[ids, 4]
        scores = -loss - .002*spread - time_weight*hours/12
        pareto = pareto_mask(np.column_stack((loss, spread, hours)))
        seen = set()
        for j in np.argsort(-scores, kind='stable'):
            i = ids[j]
            key = (round(float(p[i,0])), round(float(p[i,1]),1), int(p[i,2]), round(float(p[i,3])), round(float(p[i,4]),1))
            if key in seen:
                continue
            seen.add(key)
            within = bool(np.all(np.abs(y[i]-target) <= tolerance_fraction*np.abs(target)))
            results.append(dict(rank=len(results)+1, score=float(scores[j]), T_aus=float(p[i,0]), t_aus=float(p[i,1]),
                                quench=QUENCH_MAP[int(p[i,2])], T_temper=float(p[i,3]), t_temper=float(p[i,4]),
                                pred=dict(zip(PROPERTY_NAMES, y[i].tolist())), std=dict(zip(PROPERTY_NAMES, np.sqrt(v[i]).tolist())),
                                within_tolerance=within, pareto=bool(pareto[j]), hold_hours=float(hours[j]),
                                generator='policy' if i < count_policy else 'random'))
            if len(results) == top_k:
                break
        if not any(r['within_tolerance'] for r in results):
            warnings.append('No displayed candidate meets all target tolerances; this does not prove physical impossibility.')
    else:
        warnings.append('No candidates passed screening. No process recommendation is available.')
    return dict(schema_version='1.0', created_at=datetime.now(timezone.utc).isoformat(), material_family='Low-alloy steel',
                evidence_status='synthetic_research_only', composition_wt_pct=comp, targets=targets,
                constraints=asdict(constraints), seed=seed, method=method, candidate_budget=n_candidates,
                valid_candidates=len(ids), requested_top_k=top_k, tolerance_fraction=tolerance_fraction, time_weight=time_weight,
                model_hashes=hashes, warnings=warnings, results=results)
