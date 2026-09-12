#!/usr/bin/env python3
"""Portable, standard-library extraction of closed report evidence.

--refresh requires the original pinned JSON/documents and writes curated data.
--check verifies committed curated bytes and every original source still present.
Neither mode loads model code, checkpoints, training examples, oracle or Native.
Figures consume report-data.json directly; a clean clone needs no original runs.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import types

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V5 = 'results/factual-causal-v5-measurement-20260912/causal-v5-measurement-001'
V6 = 'results/factual-causal-v6-control-contract-preparation-20260912'
REPORT = 'results/factual-causal-v5-measurement-report-20260912/causal-v5-measurement-001'
PREP = 'results/factual-causal-v5-measurement-preparation-20260912'
POSITIONS = ['identity', 'heldout-shift8', 'heldout-gaps4']
STRATA = ['common', 'rare', 'unexposed', 'overall']
INPUTS = {
    'reference_admission': ('results/factual-canonical-control-20260912/evaluation-recovery-postrun-001/score/admission.json',
        'bdb63840ae28176e5cbf22aded90ad78585804415e0d5579ca9bb63da1bee9db'),
    'request': (PREP+'/measurement-request-001.json', '96f437b5d17392b4a406ac4336c2b2a6651924c8f30b58aeb0002d5e3eb88321'),
    'worker': (V5+'/result.json', 'f223ec5ec1101a67528a4cd54d38ce9e4d7dd405ad0c26586f63c360d42810a2'),
    'supervisor': (V5+'-supervision/terminal.json', '18b41a1980e6eef486a1747789cbb04b06154f403da7d52de9780a968acc6002'),
    'selection': (V5+'/selection.json', 'ef1b3cbb783364b662974c0a605b42a9758433b2b77c864a644d08f1d6bfa78f'),
    'review': (REPORT+'/review.json', 'ea3cbc0053cab030bc10a83253162975ab0a431eae92dde024756e5a9ae2b1ec'),
    'inventory': (V5+'/inventory.json', 'b8ffea2ff29a3543c96bdadb560c5082fd350404f4a46d46ef35ce8c98756c02'),
    'exposure': (V5+'/exposure.json', '0b31106d426f9e7da5c14dd02dac5f7bd6c2dca62062e9f1198156596e3c2c6e'),
    'archive': (V5+'/archived-reference.json', 'd4c18e613237228240274e6745a4ac112023ab167ff2cabde4c32c71c3f3e02c'),
    'persistence': (V5+'/persistence-result.json', 'f0e6e4acf44b6838c9ff3a55a8de7cc308bff66cfd1b05c410f70d713f755138'),
    'raw_readback': (V5+'/raw-readback.json', '22615cfe62401aea96ddc1a66ab8547a77490cc2745009dc4180a2ca92bc4f1c'),
    'raw_final_custody': (V5+'/raw-final-custody.json', 'ae9123b5c4aeb48ca09061633edad54c0498a43d83972deee631056bad9c155b'),
    'geometry': (V5+'/geometry.json', '9b48bba093047f9a2a07f7308f4a977100ac9914f791f924a14c8cc7ad752e14'),
    'reference_behavior': (V5+'/reference-behavior.json', 'c89bd7fd318114be789ef47346a5b9939183e83660a1cf07f3a1dc6a276116d5'),
    'v6_contract': (V6+'/control-contract.json', '0133bc4579a2c07e84752005ab8d675bac1171221f2becf2805a934f83a789b3'),
    'v6_plan': (V6+'/schedule-plan.json', 'feb9d0cbb236831a864a4f8981772262953f198fa53570ecde01b3704c95f766'),
    'v6_audit': (V6+'/design-audit.json', 'c842b43daaeab5ace86dc7693a16408fe48256a3eb37a8d1d95ac09d775bb609'),
    'v6_validation': (V6+'/preparation-validation.json', 'f4baadd6cfd87af1256fab84d5760fe1942f86dcd283ee78eaaebc1703df6353'),
    'v6_package': (V6+'/package-final.json', '80220fb1a2251c6b698015b35b94749f5c7ff4b5f0c19494fdedb7d1d00577a4'),
}
SELECTION_COPIES = {
    'causal_selection.py': ('.worktrees/factual-causal-v5/python/hytorch/causal_selection.py',
        '55445a19d520528618cf6e1d2264c66faffdadbb012e20e35e56c762a2c86b3e'),
    'FACTUAL-CAUSAL-ADMISSION-V1.md': ('docs/FACTUAL-CAUSAL-ADMISSION-V1.md',
        'a9f1aef9f00e5bb98aa7fe9e294771f279c0693a9815ae40a1ea34167d469f79'),
    'FACTUAL-CAUSAL-PILOT-PROTOCOL.md': ('docs/FACTUAL-CAUSAL-PILOT-PROTOCOL.md',
        'b5a564a0e9ddd28c29abb608ab2a0df705785f440ba4bc327002811405c2f46d'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def pretty(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def floating(token):
        value = float(token)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_float=floating,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: '+value)))


def relative_path(value):
    """Normalize historical machine-local bindings without preserving their prefix."""
    if value.startswith('/'):
        require('/results/' in value, 'unexpected historical artifact namespace')
        value = 'results/'+value.split('/results/', 1)[1]
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts, 'repository-relative path required')
    return str(path)


class Sources:
    def __init__(self, repo):
        self.repo, self.records, self.raw = Path(repo), {}, {}

    def read(self, source_id, path, expected_sha256):
        path = relative_path(path)
        require(re.fullmatch('[0-9a-f]{64}', expected_sha256), 'explicit source hash required')
        local = self.repo/path
        require(local.is_file() and local.stat().st_size <= 16*1024*1024,
                'missing or oversized closed source: '+path)
        raw = local.read_bytes()
        require(sha(raw) == expected_sha256, 'source SHA256 mismatch: '+path)
        self.records[source_id] = {'source_id':source_id, 'path':path,
            'original_sha256':expected_sha256, 'n_bytes':len(raw)}
        self.raw[source_id] = raw
        return parse(raw) if local.suffix == '.json' else None


def compressed(raw):
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0, compresslevel=9) as output:
        output.write(raw)
    return stream.getvalue()


def recompute_selection(artifacts):
    """Execute only the exact copied stdlib selector on portable reduced tables."""
    source = artifacts['causal_selection.py']
    require(sha(source) == SELECTION_COPIES['causal_selection.py'][1], 'portable selector differs from frozen source')
    for name in ('FACTUAL-CAUSAL-ADMISSION-V1.md','FACTUAL-CAUSAL-PILOT-PROTOCOL.md'):
        require(sha(artifacts[name]) == SELECTION_COPIES[name][1], 'portable rule/protocol differs from original')
    with gzip.GzipFile(fileobj=io.BytesIO(artifacts['selection-tables.json.gz']),mode='rb') as stream:
        raw = stream.read(16*1024*1024+1)
    require(len(raw) <= 16*1024*1024, 'portable selection tables exceed bound')
    tables, record = parse(raw), parse(artifacts['selection-result.json'])
    portable(tables); portable(record)
    module = types.ModuleType('_portable_causal_selection')
    module.__file__ = 'causal_selection.py'
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    checked = module.verify_selection(tables,record)
    require(checked['result']['selection_status'] == 'no-control-match'
        and checked['result']['target_unit'] is checked['result']['control_unit'] is None,
        'portable selector changed the historical failed match')
    return {'selection_status':checked['result']['selection_status'],
        'selection_internal_sha256':checked['sha256'],
        'tables_canonical_sha256':sha(canonical(tables)),
        'tables_uncompressed_bytes':len(raw), 'tables_uncompressed_sha256':sha(raw),
        'scope':'recomputed reductions, numerical-repeat checks and unchanged selection from reduced tables; no neural/raw replay',
        'raw_capture_bytes_reverified':False,'model_forwards':0,'native_queries':0}, checked


def check_curated_selection(data, record):
    """Keep plots tied to the independently reproducible portable selection."""
    selected, measured = record['result'], data['measurement']
    batteries = []
    for index,battery in enumerate(measured['batteries'],1):
        require(battery['battery'] == index, 'curated battery order differs')
        units = copy.deepcopy(battery['units'])
        for unit in units:
            for row in unit['positions']:
                m = math.sqrt(row['applied_sum_squares']/row['element_count'])
                r = math.sqrt(row['residual_sum_squares']/row['element_count'])
                derived = {'g':row['general_nll_veto']-row['general_nll_healthy'],
                    'm':m,'r':r,'rho':m/r,'p':row['admitted']/row['total_candidates']}
                for key,value in derived.items():
                    require(row.pop(key) == value, 'derived position metric differs: '+key)
        batteries.append(units)
    require(canonical(batteries) == canonical(selected['batteries']), 'curated unit/fact/position tables differ from selection')
    for output,key in (('matching','matching'),('rankings','localizer_rankings')):
        rows = [{k:v for k,v in row.items() if k != 'battery'} for row in measured[output]]
        require(canonical(rows) == canonical(selected[key]), 'curated '+output+' differs from selection')
    for key in ('selection_status','target_unit','control_unit','tentative_targets','positive_floor_passes','rule','repeat_checks'):
        require(canonical(measured[key]) == canonical(selected[key]), 'curated result differs: '+key)
    require(canonical(measured['cardinality']) == canonical(selected['declared_cardinality']), 'curated cardinality differs')


def competence_row(position, stratum, value):
    counts = {name:value[source] for name,source in (
        ('facts','facts'), ('queries','queries'), ('correct','correct'),
        ('false_assertions','false_assertion'), ('abstentions','abstention'),
        ('invalid','invalid_outputs'), ('missing','missing_predictions'))}
    require(sum(counts[k] for k in ('correct','false_assertions','abstentions','invalid','missing')) == counts['queries'],
            'classification counts do not close')
    return {'position':position, 'stratum':stratum, **counts,
        'accuracy':value['macro_fact_accuracy'], 'strict_accuracy':value['strict_accuracy'],
        'false_assertion_rate':counts['false_assertions']/counts['queries'], 'coverage':value['coverage']}


def portable(value):
    """Reject accidental inclusion of local paths, addresses, answers or token arrays."""
    if isinstance(value, dict):
        require(not set(value).intersection({'tokens','input_ids','target_ids','answer','prediction_text','prediction_tokens'}),
                'raw text/token field in portable data')
        for child in value.values():
            portable(child)
    elif isinstance(value, list):
        for child in value:
            portable(child)
    elif isinstance(value, str):
        require(not value.startswith('/') and '/home/' not in value and '/tmp/' not in value,
                'machine-local path in portable export')
        require(not re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', value), 'IP-like address in export')
    elif isinstance(value, float):
        require(math.isfinite(value), 'nonfinite exported value')


def build(repo):
    sources = Sources(repo)
    x = {name:sources.read(name,*pin) for name,pin in INPUTS.items()}
    request, worker, supervisor, review = (x[k] for k in ('request','worker','supervisor','review'))
    archive, inventory = x['archive'], x['inventory']['inventory']
    require(worker['request']['sha256'] == INPUTS['request'][1] == review['request_sha256'], 'request join failed')
    require(worker['stages']['selection']['sha256'] == INPUTS['selection'][1], 'selection join failed')
    require(any(row['sha256'] == INPUTS['selection'][1] and relative_path(row['path']) == INPUTS['selection'][0]
                for row in review['artifact_bindings']), 'review does not bind selection')
    for name in ('inventory','exposure','archive','persistence','raw_readback','raw_final_custody','geometry','reference_behavior'):
        require(worker['stages'][name]['sha256'] == INPUTS[name][1], 'worker source join failed: '+name)
    require(worker['outcome'] == 'measurement-and-selection-complete' and supervisor['reason'] == 'complete'
        and supervisor['group_empty'] is True and supervisor['runner_returncode'] == 0, 'measurement did not close')
    require(review['outcome'] == 'final-receipts-and-selection-consistent'
        and review['selection_status'] == worker['selection_status'] == 'no-control-match'
        and review['target_unit'] is review['control_unit'] is None, 'historical match was reassigned')
    raw_readback, custody = x['raw_readback'], x['raw_final_custody']
    snapshot = raw_readback['custody_snapshot']
    require(raw_readback['outcome'] == 'consistent' and custody['outcome'] == 'metadata-unchanged'
        and raw_readback['capture_transcript_head'] == custody['capture_transcript_head'] == review['capture_transcript_head']
        and custody['snapshot_id'] == snapshot['snapshot_id'] == review['raw_custody_snapshot_id'], 'closed raw/custody joins failed')
    require(custody['file_count'] == snapshot['file_count'] == len(snapshot['files'])
        and custody['directory_count'] == len(snapshot['directories']), 'closed custody namespace counts differ')
    geometry, behavior = x['geometry'], x['reference_behavior']
    qualification_pin = geometry['qualification']
    require(qualification_pin == behavior['current_geometry_report'], 'geometry/behavior report binding differs')
    qualification = sources.read('geometry_qualification',qualification_pin['path'],qualification_pin['sha256'])
    require(qualification['outcome'] == 'verified' and qualification['measurement_geometry_qualified'] is True
        and geometry['measurement_geometry_qualified'] is True, 'closed geometry was not qualified')
    same_geometry = qualification['same_geometry_inputs']
    require(len(same_geometry) == behavior['same_geometry_inputs'] == request['expected_population']['distinct_inputs']
        and all(row['all_full_logits_and_local_states_equal'] is True for row in same_geometry), 'same-geometry summary differs')
    cross_geometry = [{'position':position,**qualification['cross_geometry_diagnostic'][position]} for position in POSITIONS]
    require(sum(row['archived_forwards'] for row in cross_geometry) == behavior['archived_prefixes']
        and sum(row['argmax_changes_from_archived_token'] for row in cross_geometry) == behavior['argmax_changes'],
        'cross-geometry/behavior counts differ')
    for name,pin in request['external_modules'].items():
        sources.read('measurement_module:'+name, pin['path'], pin['sha256'])
    end_pin = x['persistence']['terminal']
    require(end_pin['path'] == 'END.json', 'unexpected capture END name')
    end = sources.read('capture_end',V5+'/captures/END.json',end_pin['sha256'])
    assembly_pin = end['assembly']
    require(assembly_pin['path'] == 'assembly.json', 'unexpected assembly name')
    assembly = sources.read('assembly',V5+'/captures/assembly.json',assembly_pin['sha256'])
    for source_id in ('capture_end','assembly'):
        source = sources.records[source_id]
        require(any(row['sha256'] == source['original_sha256'] and relative_path(row['path']) == source['path']
            for row in review['artifact_bindings']), 'final review does not bind '+source_id)
    require(sha(canonical(assembly['tables'])) == assembly['tables_sha256'], 'closed table hash mismatch')
    selection_artifacts = {'selection-tables.json.gz':compressed(canonical(assembly['tables'])+b'\n'),
        'selection-result.json':pretty(x['selection']['result']['selection'])}
    for filename,pin in SELECTION_COPIES.items():
        source_id = 'selection_copy:'+filename
        sources.read(source_id,*pin)
        selection_artifacts[filename] = sources.raw[source_id]
    selection_check, selected = recompute_selection(selection_artifacts)
    for field in ('batteries','matching','localizer_rankings','repeat_checks','rule','selection_status','target_unit','control_unit'):
        require(canonical(selected['result'][field]) == canonical(review[field]), 'portable selection differs from curated source: '+field)
    require(archive['admission']['sha256'] == INPUTS['reference_admission'][1], 'reference admission join failed')
    sources.read('reference_training_verification', archive['training_report']['path'], archive['training_report']['sha256'])

    admission, reference = x['reference_admission'], x['reference_admission']['measurement']
    require(admission['operational_verified'] is admission['scientific_admission'] is True
        and admission['causal_admission'] is False, 'reference scope changed')
    competence, priors = [], []
    for position in POSITIONS:
        metrics = reference['evaluations'][position]['metrics']
        for stratum in STRATA:
            value = metrics['overall'] if stratum == 'overall' else metrics['exposure'][stratum]
            competence.append(competence_row(position,stratum,value))
        for key in ('facts','queries','correct','false_assertion','abstention','invalid_outputs','missing_predictions'):
            require(sum(metrics['exposure'][s][key] for s in STRATA[:-1]) == metrics['overall'][key],
                    'strata do not sum to overall: '+key)
    for check in reference['numeric_gate']['checks']:
        for name,accuracy in check['baseline_macro_accuracies'].items():
            m = reference['generation_baselines'][name]['exposure']['common']
            require(accuracy == m['macro_fact_accuracy'], 'prior gate/metrics mismatch')
            priors.append({'position':check['position'],'prior':name,'common_accuracy':accuracy,
                'facts':m['facts'],'queries':m['queries'],'correct':m['correct']})

    batteries = []
    for index,units in enumerate(review['batteries'],1):
        copied = copy.deepcopy(units)
        for unit in copied:
            require(len(unit['fact_effects']) == 19 and len(unit['positions']) == 3, 'measurement membership changed')
            for row in unit['positions']:
                row['g'] = row['general_nll_veto']-row['general_nll_healthy']
                row['m'] = math.sqrt(row['applied_sum_squares']/row['element_count'])
                row['r'] = math.sqrt(row['residual_sum_squares']/row['element_count'])
                row['rho'] = row['m']/row['r']
                row['p'] = row['admitted']/row['total_candidates']
        batteries.append({'battery':index,'units':copied})
    require(len(batteries) == 2 and all([u['unit'] for u in b['units']] == [0,1,2,3] for b in batteries),
            'complete two-battery four-unit measurement required')

    contract, plan, design = x['v6_contract'], x['v6_plan'], x['v6_audit']
    require(design['contract_sha256'] == INPUTS['v6_contract'][1]
        and contract['schedule']['file_sha256'] == INPUTS['v6_plan'][1]
        and contract['schedule']['plan_sha256'] == plan['plan_sha256'] == design['plan_sha256'], 'V6 joins failed')
    require(sha(canonical({k:v for k,v in plan.items() if k != 'plan_sha256'})) == plan['plan_sha256'], 'V6 plan internal hash failed')
    require(plan['scope'] == 'design-only' and len(plan['arms']) == 17 and plan['horizon'] == 4000
        and plan['rescue_step'] == 2000 and design['world_generated'] is False, 'V6 scope changed')
    for name in ('runtime_qualified','empirical_control_validated','scientific_admission','causal_admission','training_authorized'):
        require(contract['authority'][name] is False, 'V6 acquired empirical authority')
    for filename in ('PROTOCOLO.md','control_schedule.py','review_control_design.py'):
        pin = x['v6_package']['files'][filename]
        sources.read('v6_source:'+filename, V6+'/'+filename, pin['sha256'])
    for filename,key in (('control-contract.json','v6_contract'),('schedule-plan.json','v6_plan'),('design-audit.json','v6_audit')):
        require(x['v6_package']['files'][filename]['sha256'] == INPUTS[key][1], 'V6 package mismatch')
    cost = contract['cost']
    sources.read('historical_cost_proposal', 'results/factual-canonical-control-20260912/causal-v5-preparation-v1/proposed-contract.json', cost['cost_proposal_sha256'])
    for name,pin in cost['source_terminal_pins'].items():
        terminal = sources.read('cost_terminal:'+name,pin['path'],pin['sha256'])
        require(terminal['reason'] == 'complete' and terminal['runner_returncode'] == 0
            and terminal['elapsed_seconds'] == cost['source_seconds'][name], 'historical cost terminal mismatch')
    cost_fields = ('scope','source_seconds','per_branch_seconds','branches','panel_base_seconds','new_parent_seconds',
        'panel_plus_parent_base_seconds','raw_prepost_example_bytes_per_branch','raw_prepost_example_panel_bytes',
        'not_included','gpu_speedup_measured','old_window_reusable')
    examples = []
    for label,start,stop in (('first_16',0,16),('rescue_boundary',1992,2008)):
        examples.append({'label':label,'start_step':start,'stop_step_exclusive':stop,'steps':list(range(start,stop)),
            'arms':[{'arm':arm,'veto_units':plan['calendars'][arm][start:stop]} for arm in plan['arms']]})
    summaries = [{'arm':arm,'family':re.sub(r'\d+$','',arm),
        'total':design['balance']['counts'][arm], 'first_half':design['balance']['first_half_counts'][arm],
        'second_half':design['balance']['second_half_counts'][arm]} for arm in plan['arms']]
    scopes = x['exposure']['documents']['realized_exposure']['scopes']
    data = {
        'schema':'hytorch.factual-channels-portable-report.v1','schema_version':1,
        'scope':'technical report data: measured V5 reference/access results and prospective V6 design; no confirmed causal mechanism',
        'positions':POSITIONS,'strata':STRATA,
        'reference':{'split':'discovery','training_seed':reference['training_seed'],
            'benchmark_id':reference['benchmark_id'],'checkpoint_sha256':archive['checkpoint']['sha256'],
            'checkpoint_hash_scope':'retained binding only; checkpoint bytes are not an exporter input',
            'competence':competence,'priors':priors,'gate':reference['numeric_gate'],
            'competence_threshold':0.8,'gate_scope':'common-discovery only; all three positions; strict superiority over all four priors',
            'original_production_outcome':admission['original_production_outcome'],
            'evaluation_recovered':admission['evaluation_recovered'],'operational_verified':True,
            'factual_common_discovery_admitted':True,'causal_admission':False,
            'training_geometry':inventory['geometry'], 'all_phase_value_targets':inventory['all_phase_value_targets'],
            'realized_exposure':{name:{k:row[k] for k in ('completed_updates','rows_consumed','value_targets')} for name,row in scopes.items()}},
        'measurement':{'outcome':review['outcome'],'selection_status':review['selection_status'],
            'target_unit':None,'control_unit':None,'tentative_targets':review['tentative_targets'],
            'batteries':batteries,'matching':[{'battery':i,**copy.deepcopy(row)} for i,row in enumerate(review['matching'],1)],
            'rankings':[{'battery':i,**copy.deepcopy(row)} for i,row in enumerate(review['localizer_rankings'],1)],
            'positive_floor_passes':review['positive_floor_passes'],'rule':review['rule'],
            'repeat_checks':review['repeat_checks'],'cardinality':review['cardinality'],
            'population':{'batch_count':review['batch_count'],'real_coordinates':review['real_coordinates'],
                'dummy_rows':review['dummy_rows'],'distinct_inputs':request['expected_population']['distinct_inputs'],
                'localizer_base_coordinates':inventory['localization']['row_count'],
                'control_text':{k:v for k,v in inventory['control_text'].items() if k not in ('rows','occurrence_ids')}},
            'timing':{'worker_seconds':worker['elapsed_seconds'],'supervisor_seconds':supervisor['elapsed_seconds'],
                'worker_cap_seconds':request['budget']['worker_seconds'],'supervisor_cap_seconds':request['budget']['supervisor_seconds'],
                'stages_seconds':worker['stage_timings'],'started_utc':worker['started_utc'],'finished_utc':worker['finished_utc']},
            'custody':{'native_records':review['native_records'],'final_report_json_bytes_read':review['json_bytes_read'],
                'final_report_bindings':len(review['artifact_bindings']),'capture_transcript_head':review['capture_transcript_head'],
                'raw_custody_snapshot_id':review['raw_custody_snapshot_id'],
                'payload_bytes_read':raw_readback['payload_bytes_read'],
                'file_count':custody['file_count'],'directory_count':custody['directory_count'],
                'raw_readback_outcome':raw_readback['outcome'],
                'raw_reductions_recomputed':raw_readback['raw_reductions_recomputed'],
                'assembly_recomputed':raw_readback['assembly_recomputed'],
                'complete_namespace_checked':raw_readback['complete_namespace_checked'],
                'final_metadata_outcome':custody['outcome'],
                'final_metadata_raw_bytes_rehashed':custody['raw_bytes_rehashed'],
                'final_metadata_nll_recomputed':custody['nll_recomputed'],
                'final_metadata_model_forwards':custody['model_forwards'],
                'supervisor_reason':supervisor['reason'],'supervisor_runner_returncode':supervisor['runner_returncode'],
                'supervisor_observed_runner_returncode':supervisor['observed_runner_returncode'],
                'supervisor_group_empty':supervisor['group_empty'],'supervisor_live_member_count':len(supervisor['live_members']),
                'supervisor_terminal_membership_confirmed':supervisor['terminal_membership_confirmed'],
                'scope':'reported closed capture-reader and final metadata checks; this export does not replay raw captures',
                'payload_byte_scope':'bytes read by the closed capture verifier; not complete archive/checkpoint/ledger storage size'},
            'geometry':{'qualification_outcome':qualification['outcome'],
                'batch_size':geometry['geometry']['batch'],'sequence_length':geometry['geometry']['tokens'],
                'same_geometry_input_count':len(same_geometry),
                'same_geometry_all_full_logits_and_local_states_equal':True,
                'same_geometry_scope':'historical and causal source at the same B16 geometry and exact inputs',
                'cross_geometry_diagnostic':cross_geometry,
                'cross_geometry_scope':'B1-to-B16 diagnostic on the retained archived prefixes only; differing logits do not imply changed greedy choices',
                'reference_behavior':{key:behavior[key] for key in ('archived_prefixes','same_geometry_inputs','argmax_changes',
                    'criterion','prospective_reproduction_gate_passed','new_forward_performed_here',
                    'historical_diagnostic_promoted_in_place','prior_reports_modified')},
                'effect_metrics_computed_in_geometry_qualification':qualification['effect_metrics_computed'],
                'selection_performed_in_geometry_qualification':qualification['selection_performed'],
                'scope':'retained closed qualification metadata, not new forwards or a universal cross-geometry equality guarantee'},
            'training_updates':worker['training_updates'],'oracle_reads':worker['oracle_reads'],
            'validation_test_reads':worker['validation_test_reads'],'scientific_admission':False,'causal_admission':False},
        'design_v6':{'scope':'prospective policy design only','status':'design-reviewed-and-mechanically-verified',
            'arms':plan['arms'],'horizon':plan['horizon'],'rescue_step':plan['rescue_step'],'block_size':plan['block_size'],
            'seeds':contract['replications'][0]['seeds'],'world_generated':False,'benchmark_id':None,
            'plan_sha256':plan['plan_sha256'],'calendar_file_sha256':INPUTS['v6_plan'][1],
            'calendar_examples':examples,'arm_summaries':summaries,'execution_order':contract['execution_order'],
            'balance':{key:design['balance'][key] for key in ('per_step_site_marginals_equal','distributed_per_block_site_balance',
                'rescue_prefixes_equal','rescue_suffixes_sham','steps_checked','blocks_checked','rescue_blocks',
                'actual_candidate_denials_measured','applied_magnitude_or_loss_matched')},
            'estimands':contract['estimands'],'interpretive_loss_margin':contract['interpretive_loss_margin'],
            'cost':{key:cost[key] for key in cost_fields},'authority':contract['authority'],
            'mechanical_checks':{'scheduler_tests':x['v6_validation']['scheduler']['tests_passed'],
                'builder_checks':x['v6_validation']['builder']['checks_passed'],
                'declared_decisions':x['v6_validation']['complete_plan']['decisions_declared']}},
    }
    portable(data)
    check_curated_selection(data,selected)
    return data, sorted(sources.records.values(),key=lambda row:row['source_id']), selection_artifacts, selection_check


METHODS = {
    'reference.competence':{'source':'reference_admission','fields':'measurement.evaluations[position].metrics.exposure[stratum] or metrics.overall',
        'method':'Copy counts and recorded macro_fact_accuracy/strict_accuracy/coverage; false_assertion_rate=false_assertion/queries. No answers or classification rows exported.'},
    'reference.priors':{'source':'reference_admission','fields':'measurement.numeric_gate.checks[].baseline_macro_accuracies and measurement.generation_baselines[prior].exposure.common',
        'method':'Copy all four declared entity-free common-discovery priors and exact count denominators; no fitted prior is recomputed.'},
    'reference.realized_exposure':{'source':'exposure','fields':'documents.realized_exposure.scopes',
        'method':'Copy completed_updates, rows_consumed and value_targets for all four retained scopes. Repeated exposures are not independent facts.'},
    'measurement.batteries':{'source':'review','fields':'batteries[battery][unit]',
        'method':'Copy both full unit tables, 19 fact_effects per unit, all three positions and raw summary sums/counts. Global values and selection are unchanged.'},
    'measurement.batteries.units.positions':{'source':'review','fields':'batteries[].positions[]',
        'method':'Additional g=NLL_veto-NLL_healthy; m=sqrt(applied_sum_squares/element_count); r=sqrt(residual_sum_squares/element_count); rho=m/r; p=admitted/total_candidates. These are summary arithmetic, not new model observations.'},
    'measurement.metric_definitions':{'source':'review','fields':'rule, batteries, matching',
        'method':'Delta Value: full428 next-Value NLL difference, mean prompt/position within fact then equal facts. g: signed original valid-token NLL difference, equal maps. m/r: local RMS relative to ordinary zero-commit quantized output, global sums before sqrt. p includes magnitude-zero COMMIT and retains all healthy candidate opportunities.'},
    'measurement.matching':{'source':'review','fields':'matching, rule, localizer_rankings, repeat_checks',
        'method':'Copy all candidates, signed differences, distances, individual calipers and both numerical repetitions. No threshold adjustment, reranking, bootstrap or confidence interval.'},
    'measurement.population_timing_custody':{'source':'worker,supervisor,inventory,request,review',
        'fields':'elapsed_seconds, stage_timings, expected_population, inventory.control_text, review.cardinality and custody counters',
        'method':'Copy final complete counts and elapsed durations; repeated batteries are numerical reproducibility, not independent training replications.'},
    'measurement.custody.closed_checks':{'source':'raw_readback,raw_final_custody,supervisor',
        'fields':'raw_readback outcome/payload_bytes_read/recomputation flags; raw_final_custody outcome/file_count/directory_count/recheck flags; terminal process-group closure',
        'method':'Copy final retained summaries after hash/worker and transcript/snapshot joins. payload_bytes_read is consumed capture-reader bytes, not total archive size. The export opens JSON metadata only and does not rehash raw payloads or rerun their reductions.'},
    'measurement.geometry':{'source':'geometry,geometry_qualification,reference_behavior',
        'fields':'geometry.qualification; qualification.same_geometry_inputs and cross_geometry_diagnostic; behavior archived_prefixes/same_geometry_inputs/argmax_changes/criterion',
        'method':'Verify the closed report binding shared by geometry and behavior. Aggregate the all_equal flags over219 same-geometry inputs. Copy all three B1/B16 diagnostic rows:384 archived prefixes/map, differing full428 vectors and maximum logit difference, with no new forward; summed argmax changes/prefix counts must equal the retained behavior gate.'},
    'design_v6.calendar_examples':{'source':'v6_plan','fields':'calendars[arm][0:16] and calendars[arm][1992:2008]',
        'method':'Slice the pinned complete calendar for all17 arms. Indices are zero-based; null is sham. No new calendar or world is generated.'},
    'design_v6.arm_summaries':{'source':'v6_audit','fields':'balance.counts, first_half_counts, second_half_counts',
        'method':'Copy opportunity counts from the closed pure design audit; not measured candidate denials or matched empirical damage.'},
    'design_v6.cost':{'source':'v6_contract and eight cost_terminal sources','fields':'cost',
        'method':'Copy serial CPU extrapolation, after checking every source terminal hash/status/time. Base excludes new qualification/captures/curves/diagnostics/storage/contingency and is not a sufficient authorized cap.'},
    'portable_selection':{'source':'assembly, capture_end, persistence, selection and exact selection_copy sources',
        'fields':'assembly.tables; selection.result.selection',
        'method':'Verify worker->persistence->END->assembly and final-review links, then export canonical reduced tables as gzip (mtime=0, filename empty) and the original inner selection record. Exact stdlib selector/rule copies recompute reductions and no-control-match, without raw tensor or neural replay.'},
}


def check(repo):
    raw = (HERE/'report-data.json').read_bytes()
    provenance = parse((HERE/'provenance.json').read_bytes())
    require(sha(raw) == provenance['curated']['sha256'], 'committed curated-data hash mismatch')
    require(sha(Path(__file__).read_bytes()) == provenance['exporter']['sha256'], 'exporter source hash mismatch')
    require(sha((HERE/'verify_selection.py').read_bytes()) == provenance['portable_verifier']['sha256'], 'portable verifier source hash mismatch')
    data = parse(raw)
    portable(data); portable(provenance)
    artifacts = {}
    for name,descriptor in provenance['portable_selection']['files'].items():
        require(PurePosixPath(name).name == name, 'selection artifact must be a local basename')
        content = (HERE/name).read_bytes()
        require(sha(content) == descriptor['sha256'], 'portable selection artifact hash mismatch: '+name)
        artifacts[name] = content
    selection_check,selected = recompute_selection(artifacts)
    check_curated_selection(data,selected)
    require(canonical(selection_check) == canonical(provenance['portable_selection']['recomputation']),
            'portable selection recomputation differs')
    missing, verified = [], 0
    for source in provenance['sources']:
        path = Path(repo)/relative_path(source['path'])
        if not path.exists():
            missing.append(source['source_id']); continue
        require(path.is_file() and path.stat().st_size <= 16*1024*1024, 'unexpected source artifact size/type')
        require(sha(path.read_bytes()) == source['original_sha256'], 'available original source changed: '+source['path'])
        verified += 1
    reconstructed = False
    if not missing:
        rebuilt,_,rebuilt_artifacts,_ = build(repo)
        require(pretty(rebuilt) == raw, 'curated data differs from original reconstruction')
        require(rebuilt_artifacts == artifacts, 'portable selection differs from original reconstruction')
        reconstructed = True
    return {'outcome':'portable-data-verified','original_sources_verified':verified,
        'original_sources_missing':len(missing),'reconstructed_from_originals':reconstructed,
        'portable_selection_recomputed':True,'selection_status':selection_check['selection_status'],
        'scope':'missing originals limit provenance recheck, not figure reconstruction from committed JSON'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=REPO)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--refresh',action='store_true')
    mode.add_argument('--check',action='store_true')
    args = parser.parse_args()
    if args.check:
        result = check(args.repo_root)
    else:
        data,sources,selection_artifacts,selection_check = build(args.repo_root)
        raw = pretty(data)
        provenance = {'schema':'hytorch.portable-report-provenance.v1',
            'scope':'curated closed metadata only; no fresh measurements, raw tensor replay or causal admission',
            'curated':{'path':'reports/factual-channels-2026-09-12/data/report-data.json','sha256':sha(raw),'n_bytes':len(raw)},
            'exporter':{'path':'reports/factual-channels-2026-09-12/data/export_data.py','sha256':sha(Path(__file__).read_bytes())},
            'portable_verifier':{'path':'reports/factual-channels-2026-09-12/data/verify_selection.py',
                'sha256':sha((HERE/'verify_selection.py').read_bytes())},
            'sources':sources,'field_methods':METHODS,
            'evidence_ids':{'E01':{'source_id':'reference_admission','field':'measurement.evaluations and numeric_gate'},
                'E02':{'source_id':'worker','field':'final measurement result and stage timings'},
                'E03':{'source_id':'supervisor','field':'terminal closure and elapsed_seconds'},
                'E04':{'source_id':'selection','field':'result.selection','portable_file':'selection-result.json'},
                'E05':{'source_id':'assembly','field':'tables','portable_file':'selection-tables.json.gz'},
                'E06':{'source_id':'v6_plan','field':'complete prospective schedule; illustrative slices in report-data.json'}},
            'portable_selection':{'files':{name:{'sha256':sha(content),'n_bytes':len(content)}
                for name,content in sorted(selection_artifacts.items())},'recomputation':selection_check},
            'source_hashes_are_original_bytes':True,'curated_hash_is_distinct_from_source_hashes':True,
            'omitted':['raw tensors/checkpoints','training text and token arrays','answers and private labels',
                'Native ledgers','machine-local paths and infrastructure addresses'],
            'limitations':['source hashes authenticate the retained files, not new neural replay',
                'reference competence is common-discovery and one training seed',
                'V5 localization is access dependence; no-control-match is preserved',
                'two repeated batteries are not independent scientific replications',
                'V6 calendars are prospective and empirical_control_validated remains false']}
        portable(provenance)
        (HERE/'report-data.json').write_bytes(raw)
        for name,content in selection_artifacts.items():
            (HERE/name).write_bytes(content)
        (HERE/'provenance.json').write_bytes(pretty(provenance))
        result = {'outcome':'curated-data-exported','sha256':sha(raw),'n_bytes':len(raw),'source_count':len(sources)}
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
