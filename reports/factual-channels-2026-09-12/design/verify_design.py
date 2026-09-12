"""Verify the copied V6 schedule using only the pinned source bytes and plan."""
from pathlib import Path
from hashlib import sha256
import json
import types

EXPECTED = {
    'PROTOCOLO.md':'6bae2857c9bf6910aa728be2cd65da5f510400190f2e45a1d909194bd48ea195',
    'control_schedule.py':'7ff27c86b5da1f6b223472d9394a62f9aba1f56c4a65b19ca44057efe13916c8',
    'test_control_schedule.py':'a5754407fa8f49bd8db3b6ce5398e06014f6acc213e13f0283414945e52ff29e',
    'schedule-plan.json':'feb9d0cbb236831a864a4f8981772262953f198fa53570ecde01b3704c95f766',
}


def main():
    root = Path(__file__).resolve().parent
    provenance = json.loads((root/'provenance.json').read_text())
    if set(provenance['files']) != set(EXPECTED):
        raise ValueError('the fixed design-file membership changed')
    blobs = {}
    for name, item in provenance['files'].items():
        if Path(name).name != name:
            raise ValueError('simple file names required')
        raw = (root/name).read_bytes()
        if item['sha256'] != EXPECTED[name] or sha256(raw).hexdigest() != EXPECTED[name]:
            raise ValueError('design artifact hash mismatch: '+name)
        blobs[name] = raw
    module = types.ModuleType('_report_pinned_v6_schedule')
    module.__file__ = str(root/'control_schedule.py')
    exec(compile(blobs['control_schedule.py'], module.__file__, 'exec'), module.__dict__)
    plan = json.loads(blobs['schedule-plan.json'])
    if plan['schedule_seed'] != 2026091304 or plan['scope'] != 'design-only':
        raise ValueError('the declared prospective plan changed')
    audit = module.audit_balance(plan)
    print(json.dumps({'outcome':'prospective-design-consistent',
        'arms':audit['arm_count'],'steps':audit['steps_checked'],
        'plan_sha256':plan['plan_sha256'],'per_step_balance':audit['per_step_site_marginals_equal'],
        'per_block_balance':audit['distributed_per_block_site_balance'],
        'rescue_prefixes_equal':audit['rescue_prefixes_equal'],
        'model_forwards':0,'training_authorized':False},sort_keys=True))


if __name__ == '__main__':
    main()
