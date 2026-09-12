"""Pure schedule contracts only: no NN, Native, dataset or measurement fixture."""
import copy
import hashlib
import json
import random

import pytest

import control_schedule as schedule


SEED = 2026091304


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def reseal(plan, *, calendars=False):
    if calendars:
        for arm, values in plan['calendars'].items():
            plan['calendar_sha256'][arm] = hashlib.sha256(canonical([schedule.SCHEMA, 'calendar', arm, values])).hexdigest()
    plan['plan_sha256'] = hashlib.sha256(canonical({key: value for key, value in plan.items() if key != 'plan_sha256'})).hexdigest()
    return plan


@pytest.fixture
def tiny():
    return schedule.build_plan(SEED, 8, 4, scope='fixture')


@pytest.fixture(scope='module')
def production():
    return schedule.build_plan(SEED)


def test_known_hash_vectors_and_small_calendars(tiny):
    assert schedule._permutation_digest(SEED, 0, 0) == '1d74f541cc5893acf56f51154257f8c8a24184f4556dc8f7ae6a0c616a7d0db9'
    assert schedule._permutation_digest(SEED, 1, 3) == '0ffb5ef8c042f4fc09eca3eb6c2fad2a6760f169c67b43b9bd0a12583bbc7a42'
    assert tiny['block_permutations'] == [[0, 1, 2, 3], [3, 1, 2, 0]]
    expected = {
        'D0': [0, 1, 2, 3, 3, 1, 2, 0], 'D1': [1, 2, 3, 0, 1, 2, 0, 3],
        'D2': [2, 3, 0, 1, 2, 0, 3, 1], 'D3': [3, 0, 1, 2, 0, 3, 1, 2],
    }
    assert tiny['calendars']['H'] == [None]*8
    for unit in range(4):
        assert tiny['calendars'][f'P{unit}'] == [unit]*8
        assert tiny['calendars'][f'RP{unit}'] == [unit]*4+[None]*4
        assert tiny['calendars'][f'D{unit}'] == expected[f'D{unit}']
        assert tiny['calendars'][f'RD{unit}'] == expected[f'D{unit}'][:4]+[None]*4


def test_production_complete_balance_counts_and_rescue_halves(production):
    result = schedule.audit_balance(production)
    assert production['arms'] == ['H', 'P0', 'P1', 'P2', 'P3', 'D0', 'D1', 'D2', 'D3',
                                 'RP0', 'RP1', 'RP2', 'RP3', 'RD0', 'RD1', 'RD2', 'RD3']
    assert result['arm_count'] == 17 and result['steps_checked'] == 4000
    assert result['blocks_checked'] == 1000 and result['rescue_blocks'] == 500
    assert result['per_step_site_marginals_equal'] and result['distributed_per_block_site_balance']
    assert result['rescue_prefixes_equal'] and result['rescue_suffixes_sham']
    assert result['counts']['H'] == {'per_site': [0]*4, 'vetoed_updates': 0, 'sham_updates': 4000}
    for unit in range(4):
        persistent = [0]*4; persistent[unit] = 4000
        rescued = [0]*4; rescued[unit] = 2000
        assert result['counts'][f'P{unit}'] == {'per_site': persistent, 'vetoed_updates': 4000, 'sham_updates': 0}
        assert result['counts'][f'D{unit}'] == {'per_site': [1000]*4, 'vetoed_updates': 4000, 'sham_updates': 0}
        assert result['counts'][f'RP{unit}'] == {'per_site': rescued, 'vetoed_updates': 2000, 'sham_updates': 2000}
        assert result['counts'][f'RD{unit}'] == {'per_site': [500]*4, 'vetoed_updates': 2000, 'sham_updates': 2000}
        assert result['first_half_counts'][f'D{unit}']['per_site'] == [500]*4
        assert result['second_half_counts'][f'RD{unit}']['sham_updates'] == 2000
        matrix = result['distributed_site_by_offset_counts'][f'D{unit}']
        assert all(sum(row) == 1000 for row in matrix)
        assert [sum(row[offset] for row in matrix) for offset in range(4)] == [1000]*4
    assert result['actual_candidate_denials_measured'] is False and result['applied_magnitude_or_loss_matched'] is False


def test_audit_checks_all_steps_and_blocks_without_decision_loop(production, monkeypatch):
    original = schedule.validate_plan; calls = []
    def validate(plan):
        calls.append(True)
        return original(plan)
    monkeypatch.setattr(schedule, 'validate_plan', validate)
    monkeypatch.setattr(schedule, 'decision', lambda *a, **kw: pytest.fail('audit cannot revalidate each step through decision'))
    result = schedule.audit_balance(production)
    assert len(calls) == 1 and result['steps_checked'] == 4000
    calendars = production['calendars']
    assert all(set(row) == {0, 1, 2, 3} for row in zip(*(calendars[f'D{s}'] for s in range(4))))
    for s in range(4):
        values = calendars[f'D{s}']
        assert all(set(values[start:start+4]) == {0, 1, 2, 3} for start in range(0, 4000, 4))


def test_exact_rescue_boundary_and_primary_inference(production):
    assert production['block_permutations'][499] == [3, 0, 1, 2]
    assert production['block_permutations'][500] == [1, 3, 0, 2]
    for index in range(4):
        for family in ('P', 'D'):
            full, rescued = f'{family}{index}', f'R{family}{index}'
            before = schedule.decision(production, rescued, 'training', optimizer_step=1999)
            assert before['deny_write_units'] == [production['calendars'][full][1999]]
            assert before['block_index'] == 499 and before['index_in_block'] == 3
            assert schedule.decision(production, rescued, 'training', optimizer_step=2000)['deny_write_units'] == []
            assert schedule.decision(production, rescued, 'training', optimizer_step=3999)['deny_write_units'] == []
            assert len(schedule.decision(production, full, 'training', optimizer_step=2000)['deny_write_units']) == 1
    for arm in production['arms']:
        decision = schedule.decision(production, arm, 'inference')
        assert decision['deny_write_units'] == [] and decision['global_optimizer_step'] is None
        assert decision['block_index'] is decision['index_in_block'] is decision['token_step'] is None
        assert not decision['training_authorized'] and not decision['scientific_admission'] and not decision['causal_admission']


def test_deterministic_prefixes_seed_domain_and_no_global_rng(tiny, production):
    before = random.getstate()
    assert schedule.build_plan(SEED, 8, 4, scope='fixture') == tiny
    assert production['block_permutations'][:2] == tiny['block_permutations']
    assert schedule.build_plan(0, 8, 4, scope='fixture')['block_permutations'][0] == [3, 2, 0, 1]
    assert schedule.build_plan(2**64-1, 8, 4, scope='fixture')['schedule_seed'] == 2**64-1
    assert len({tuple(p) for p in production['block_permutations']}) > 1
    schedule.validate_plan(tiny); schedule.audit_balance(tiny); schedule.decision(tiny, 'D0', 'training', 0)
    assert random.getstate() == before


def test_hash_ties_use_unit_index(monkeypatch):
    monkeypatch.setattr(schedule, '_permutation_digest', lambda seed, block, unit: '0'*64)
    plan = schedule.build_plan(SEED, 8, 4, scope='fixture')
    assert plan['block_permutations'] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert schedule.validate_plan(plan) == plan


def test_json_roundtrip_and_private_return_values(tiny):
    roundtrip = json.loads(canonical(tiny))
    assert schedule.validate_plan(roundtrip) == tiny
    assert len(tiny['calendar_sha256']) == 17
    for arm in tiny['arms']:
        assert tiny['calendar_sha256'][arm] == hashlib.sha256(canonical([schedule.SCHEMA, 'calendar', arm, tiny['calendars'][arm]])).hexdigest()
    validated = schedule.validate_plan(tiny)
    validated['calendars']['H'][0] = 0
    assert tiny['calendars']['H'][0] is None
    d = schedule.decision(tiny, 'D0', 'training', 0); d['deny_write_units'].append(3)
    assert schedule.decision(tiny, 'D0', 'training', 0)['deny_write_units'] == [0]
    balance = schedule.audit_balance(tiny); balance['counts']['D0']['per_site'][0] = 999
    assert schedule.audit_balance(tiny)['counts']['D0']['per_site'] == [2]*4


@pytest.mark.parametrize('seed', [True, False, -1, 2**64, 1., '1', None])
def test_seed_must_be_explicit_u64_without_bool(seed):
    with pytest.raises(ValueError):
        schedule.build_plan(seed)


@pytest.mark.parametrize('horizon,rescue,scope', [
    (8, 4, 'design-only'), (4000, 1999, 'design-only'), (4000., 2000, 'design-only'),
    (4000, True, 'design-only'), (True, 4, 'fixture'), (4, 2, 'fixture'), (0, 0, 'fixture'),
    (12, 4, 'fixture'), (16, 4, 'fixture'), (8, 8, 'fixture'), (4008, 2004, 'fixture'),
    (8, 4, 'production'), (8, 4, False),
])
def test_only_explicit_small_equal_half_fixture_geometry_is_allowed(horizon, rescue, scope):
    with pytest.raises(ValueError):
        schedule.build_plan(SEED, horizon, rescue, scope=scope)


@pytest.mark.parametrize('arm,phase,step,token', [
    ('H0', 'training', 0, None), ('P4', 'training', 0, None), ('D-1', 'training', 0, None),
    ('RD04', 'training', 0, None), (True, 'training', 0, None), ('T', 'training', 0, None),
    ('H', 'train', 0, None), ('H', False, 0, None), ('H', 'training', None, None),
    ('H', 'training', True, None), ('H', 'training', 0., None), ('H', 'training', -1, None),
    ('H', 'training', 8, None), ('H', 'training', 0, 0), ('H', 'inference', 0, None),
    ('H', 'inference', False, None), ('H', 'inference', None, 0),
])
def test_phase_coordinates_and_arm_names_are_strict(tiny, arm, phase, step, token):
    with pytest.raises(ValueError):
        schedule.decision(tiny, arm, phase, step, token)


@pytest.mark.parametrize('change', [
    'calendar', 'rescue_prefix', 'rescue_boundary', 'missing_arm', 'extra_arm', 'arm_order',
    'bool_unit', 'tuple_calendar', 'permutation', 'seed', 'hash', 'unknown_field', 'training_flag',
    'scientific_flag', 'compatible_flag', 'primary_veto',
])
def test_mutated_serialization_is_rejected_even_with_repaired_hashes(tiny, change):
    plan = json.loads(canonical(tiny))
    if change == 'calendar': plan['calendars']['D0'][0] = 1
    elif change == 'rescue_prefix': plan['calendars']['RP0'][0] = 1
    elif change == 'rescue_boundary': plan['calendars']['RD0'][4] = 3
    elif change == 'missing_arm': plan['calendars'].pop('D3')
    elif change == 'extra_arm': plan['calendars']['C'] = [0]*8
    elif change == 'arm_order': plan['arms'][0], plan['arms'][1] = plan['arms'][1], plan['arms'][0]
    elif change == 'bool_unit': plan['calendars']['P0'][0] = False
    elif change == 'tuple_calendar': plan['calendars']['P0'] = tuple(plan['calendars']['P0'])
    elif change == 'permutation': plan['block_permutations'][0] = [1, 0, 2, 3]
    elif change == 'seed': plan['schedule_seed'] += 1
    elif change == 'hash': plan['calendar_sha256']['P0'] = '0'*64
    elif change == 'unknown_field': plan['override'] = True
    elif change == 'training_flag': plan['training_authorized'] = True
    elif change == 'scientific_flag': plan['scientific_admission'] = True
    elif change == 'compatible_flag': plan['frozen_v5_execution_compatible'] = True
    elif change == 'primary_veto': plan['primary_inference']['deny_write_units'] = [0]
    reseal(plan, calendars=change != 'hash')
    with pytest.raises(ValueError):
        schedule.validate_plan(plan)
    with pytest.raises(ValueError):
        schedule.audit_balance(plan)


def test_balanced_but_wrong_hash_permutation_is_not_a_valid_seed_plan(tiny):
    plan = copy.deepcopy(tiny)
    changed = [1, 0, 3, 2]
    plan['block_permutations'][0] = changed
    for offset in range(4):
        values = [changed[(step+offset) % 4] for step in range(4)]
        plan['calendars'][f'D{offset}'][:4] = values
        plan['calendars'][f'RD{offset}'][:4] = values
    reseal(plan, calendars=True)
    assert all(set(row) == {0, 1, 2, 3} for row in zip(*(plan['calendars'][f'D{s}'] for s in range(4))))
    with pytest.raises(ValueError, match='deterministic reconstruction'):
        schedule.validate_plan(plan)


def test_all_outputs_keep_design_scope_and_no_admission(production, tiny):
    assert production['scope'] == 'design-only' and tiny['scope'] == 'fixture'
    assert production['frozen_v5_execution_compatible'] is False
    assert production['requires_new_execution_qualification'] is True
    assert production['model_rng_consumed'] is False
    for value in (production, tiny, schedule.audit_balance(tiny), schedule.decision(tiny, 'P0', 'training', 0)):
        assert value['training_authorized'] is value['scientific_admission'] is value['causal_admission'] is False
