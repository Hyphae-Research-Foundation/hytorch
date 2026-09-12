"""Pure prospective persistence-versus-distribution schedule, not an executor.

All outputs are declarations. No model, RNG, clock, file or Native API is used.
Balance counts whole-unit veto opportunities, not actual surviving COMMITs,
applied magnitude, loss, or equivalent general disruption. Frozen V5 execution
does not accept this new schedule family without explicit new qualification.
"""
from __future__ import annotations

import copy
import hashlib
import json

SCHEMA = 'hytorch.causal-v6-persistence-schedule.v1'
DECISION_SCHEMA = 'hytorch.causal-v6-persistence-decision.v1'
AUDIT_SCHEMA = 'hytorch.causal-v6-persistence-balance.v1'
UNITS = (0, 1, 2, 3)
ARMS = ('H', *(f'P{u}' for u in UNITS), *(f'D{s}' for s in UNITS),
        *(f'RP{u}' for u in UNITS), *(f'RD{s}' for s in UNITS))
PERMUTATION_RULE = "lexicographic SHA256(canonical JSON [schema,'permutation',schedule_seed,block,unit]); ties by unit"
CALENDAR_RULE = "P_u(t)=u; D_s(t)=pi[t//4][(t%4+s)%4]; RP/RD same prefix and sham from rescue; H always sham"
_FIELDS = {'schema', 'scope', 'schedule_seed', 'horizon', 'rescue_step', 'units', 'arms',
    'block_size', 'permutation_rule', 'calendar_rule', 'canonical_json', 'block_permutations',
    'calendars', 'calendar_sha256', 'primary_inference', 'dose_unit',
    'model_rng_consumed', 'frozen_v5_execution_compatible', 'requires_new_execution_qualification',
    'training_authorized', 'scientific_admission', 'causal_admission', 'plan_sha256'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _parameters(schedule_seed, horizon, rescue_step, scope):
    _require(type(schedule_seed) is int and 0 <= schedule_seed < 2**64,
             'schedule_seed must be an explicit unsigned 64-bit integer, never bool')
    _require(type(scope) is str and scope in ('design-only', 'fixture'), 'explicit design-only or fixture scope required')
    _require(type(horizon) is int and type(rescue_step) is int,
             'horizon and rescue_step must be exact integers, never bool')
    if scope == 'design-only':
        _require((horizon, rescue_step) == (4000, 2000), 'production design fixes horizon4000/rescue2000')
    else:
        _require(8 <= horizon <= 4000 and horizon % 8 == 0 and rescue_step == horizon//2,
                 'fixture needs equal positive halves, each on a complete four-step boundary')


def _permutation_digest(schedule_seed, block, unit):
    return _sha([SCHEMA, 'permutation', schedule_seed, block, unit])


def _calendar_hash(arm, calendar):
    return _sha([SCHEMA, 'calendar', arm, calendar])


def _build(schedule_seed, horizon, rescue_step, scope):
    permutations = [sorted(UNITS, key=lambda u: (_permutation_digest(schedule_seed, block, u), u))
                    for block in range(horizon//4)]
    calendars = {'H': [None]*horizon}
    for unit in UNITS:
        calendars[f'P{unit}'] = [unit]*horizon
    for offset in UNITS:
        calendars[f'D{offset}'] = [permutations[step//4][(step % 4+offset) % 4] for step in range(horizon)]
    for family in ('P', 'D'):
        for index in UNITS:
            calendars[f'R{family}{index}'] = calendars[f'{family}{index}'][:rescue_step]+[None]*(horizon-rescue_step)
    body = {'schema': SCHEMA, 'scope': scope, 'schedule_seed': schedule_seed, 'horizon': horizon,
        'rescue_step': rescue_step, 'units': list(UNITS), 'arms': list(ARMS), 'block_size': 4,
        'permutation_rule': PERMUTATION_RULE, 'calendar_rule': CALENDAR_RULE,
        'canonical_json': 'UTF-8; sorted object keys; compact separators; ensure_ascii=false; no NaN/Infinity',
        'block_permutations': permutations, 'calendars': calendars,
        'calendar_sha256': {arm: _calendar_hash(arm, calendars[arm]) for arm in ARMS},
        'primary_inference': {'deny_write_units': [], 'optimizer_step': None, 'token_step': None},
        'dose_unit': 'whole-unit veto opportunities per training forward; actual COMMITs/magnitude/loss are not matched',
        'model_rng_consumed': False, 'frozen_v5_execution_compatible': False,
        'requires_new_execution_qualification': True, 'training_authorized': False,
        'scientific_admission': False, 'causal_admission': False}
    return {**body, 'plan_sha256': _sha(body)}


def build_plan(schedule_seed, horizon=4000, rescue_step=2000, *, scope='design-only'):
    """Build all 17 complete calendars; small horizons require explicit fixture scope."""
    _parameters(schedule_seed, horizon, rescue_step, scope)
    return _build(schedule_seed, horizon, rescue_step, scope)


def validate_plan(plan):
    """Reconstruct from declared seed and compare every field, returning a private copy.

    A repaired self-hash cannot substitute a different balanced permutation or
    calendar. This validates a design, not externally approved plan identity or
    execution authority. A caller must separately pin the returned plan SHA.
    """
    _require(type(plan) is dict and set(plan) == _FIELDS, 'complete exact schedule plan fields required')
    _parameters(plan['schedule_seed'], plan['horizon'], plan['rescue_step'], plan['scope'])
    horizon = plan['horizon']
    _require(type(plan['arms']) is list and plan['arms'] == list(ARMS)
        and type(plan['units']) is list and all(type(u) is int for u in plan['units'])
        and plan['units'] == list(UNITS), 'canonical ordered membership of all17 arms and four units required')
    permutations = plan['block_permutations']
    _require(type(permutations) is list and len(permutations) == horizon//4,
             'one complete permutation for every four-step block required')
    for permutation in permutations:
        _require(type(permutation) is list and len(permutation) == 4
            and all(type(u) is int for u in permutation) and sorted(permutation) == list(UNITS),
            'each block must be a permutation of four exact integer units')
    calendars = plan['calendars']
    _require(type(calendars) is dict and set(calendars) == set(ARMS), 'all17 exact arm calendars required')
    for calendar in calendars.values():
        _require(type(calendar) is list and len(calendar) == horizon
            and all(unit is None or type(unit) is int and unit in UNITS for unit in calendar),
            'every step must declare sham or exactly one integer unit, never a candidate mask')
    _require(type(plan['calendar_sha256']) is dict and set(plan['calendar_sha256']) == set(ARMS),
             'every complete arm calendar needs its own digest')
    expected = _build(plan['schedule_seed'], horizon, plan['rescue_step'], plan['scope'])
    _require(_canonical(plan) == _canonical(expected), 'schedule differs from exact deterministic reconstruction or its hashes')
    return copy.deepcopy(expected)


def decision(plan, arm, phase, optimizer_step=None, token_step=None):
    """Return a new []/[unit] declaration; generation coordinates never select training."""
    checked = validate_plan(plan)
    _require(type(arm) is str and arm in ARMS, 'one exact declared arm name required')
    _require(type(phase) is str and phase in ('training', 'inference'), 'explicit training or inference phase required')
    _require(token_step is None, 'token_step cannot select this optimizer schedule')
    if phase == 'inference':
        _require(optimizer_step is None, 'primary inference cannot carry an optimizer step')
        unit = None
    else:
        _require(type(optimizer_step) is int and 0 <= optimizer_step < checked['horizon'],
                 'bounded zero-based global optimizer step required, never bool')
        unit = checked['calendars'][arm][optimizer_step]
    return {'schema': DECISION_SCHEMA, 'scope': checked['scope'], 'plan_sha256': checked['plan_sha256'],
        'calendar_sha256': checked['calendar_sha256'][arm], 'arm': arm, 'phase': phase,
        'global_optimizer_step': optimizer_step, 'token_step': None,
        'block_index': optimizer_step//4 if phase == 'training' else None,
        'index_in_block': optimizer_step % 4 if phase == 'training' else None,
        'deny_write_units': [] if unit is None else [unit],
        'training_authorized': False, 'scientific_admission': False, 'causal_admission': False}


def _counts(calendar):
    return {'per_site': [calendar.count(unit) for unit in UNITS],
            'vetoed_updates': sum(unit is not None for unit in calendar), 'sham_updates': calendar.count(None)}


def audit_balance(plan):
    """Exhaustively check unit-opportunity counts, all steps/blocks and rescue prefixes.

    The phase tables expose finite associations with step%4; hash permutation
    does not promise exact unit-by-offset balance within each distributed arm.
    No count here measures actual removed writes, magnitude or model effects.
    """
    checked = validate_plan(plan); calendars = checked['calendars']
    horizon, rescue = checked['horizon'], checked['rescue_step']
    for step in range(horizon):
        _require(sorted(calendars[f'P{u}'][step] for u in UNITS) == list(UNITS)
            and sorted(calendars[f'D{s}'][step] for s in UNITS) == list(UNITS),
            'persistent/distributed site marginals must match exactly at each step')
        if step < rescue:
            _require(sorted(calendars[f'RP{u}'][step] for u in UNITS) == list(UNITS)
                and sorted(calendars[f'RD{s}'][step] for s in UNITS) == list(UNITS),
                'rescue families must retain the same pre-rescue site marginals')
        else:
            _require(all(calendars[f'{family}{u}'][step] is None for family in ('RP', 'RD') for u in UNITS),
                     'all rescue arms must be sham from the declared boundary')
    for arm in (f'D{s}' for s in UNITS):
        for start in range(0, horizon, 4):
            _require(sorted(calendars[arm][start:start+4]) == list(UNITS), 'each distributed arm needs every site once per block')
    for family in ('P', 'D'):
        for index in UNITS:
            _require(calendars[f'R{family}{index}'][:rescue] == calendars[f'{family}{index}'][:rescue],
                     'each rescue prefix must equal its corresponding full-treatment prefix')
    counts = {arm: _counts(calendars[arm]) for arm in ARMS}
    phase_counts = {f'D{s}': [[sum(calendars[f'D{s}'][step] == unit for step in range(offset, horizon, 4))
                              for offset in UNITS] for unit in UNITS] for s in UNITS}
    body = {'schema': AUDIT_SCHEMA, 'scope': checked['scope'], 'plan_sha256': checked['plan_sha256'],
        'arm_count': 17, 'steps_checked': horizon, 'blocks_checked': horizon//4, 'rescue_blocks': rescue//4,
        'per_step_site_marginals_equal': True, 'distributed_per_block_site_balance': True,
        'rescue_prefixes_equal': True, 'rescue_suffixes_sham': True,
        'counts': counts, 'first_half_counts': {arm: _counts(calendars[arm][:rescue]) for arm in ARMS},
        'second_half_counts': {arm: _counts(calendars[arm][rescue:]) for arm in ARMS},
        'distributed_site_by_offset_counts': phase_counts,
        'offset_table_axes': ['site', 'optimizer_step_mod4'],
        'actual_candidate_denials_measured': False, 'applied_magnitude_or_loss_matched': False,
        'training_authorized': False, 'scientific_admission': False, 'causal_admission': False}
    return {**body, 'audit_sha256': _sha(body)}
