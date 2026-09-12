#!/usr/bin/env python3
"""Reproduce the complete preserved selection from local portable files only.

Standard library only. No original results directory, model, labels or Native.
The reduced tables are evidence inputs; this does not verify their raw captures.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import types

FILES = {'selection-tables.json.gz', 'selection-result.json', 'causal_selection.py',
         'FACTUAL-CAUSAL-ADMISSION-V1.md', 'FACTUAL-CAUSAL-PILOT-PROTOCOL.md'}
SELECTOR_SHA256 = '55445a19d520528618cf6e1d2264c66faffdadbb012e20e35e56c762a2c86b3e'
RULE_SHA256 = 'a9f1aef9f00e5bb98aa7fe9e294771f279c0693a9815ae40a1ea34167d469f79'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    def floating(token):
        value = float(token)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    return json.loads(raw,object_pairs_hook=pairs,parse_float=floating,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: '+value)))


def verify(data_dir):
    root = Path(data_dir)
    manifest = parse((root/'provenance.json').read_bytes())
    require(sha(Path(__file__).read_bytes()) == manifest['portable_verifier']['sha256'], 'verifier hash mismatch')
    expected = manifest['portable_selection']
    require(set(expected['files']) == FILES, 'complete portable selection files required')
    artifacts = {}
    for name,descriptor in expected['files'].items():
        path = root/name
        require(path.is_file() and path.stat().st_size <= 16*1024*1024, 'bounded portable artifact required')
        raw = path.read_bytes()
        require(sha(raw) == descriptor['sha256'] and len(raw) == descriptor['n_bytes'], 'artifact hash/size mismatch: '+name)
        artifacts[name] = raw
    require(sha(artifacts['causal_selection.py']) == SELECTOR_SHA256, 'selector differs from frozen pure implementation')
    require(sha(artifacts['FACTUAL-CAUSAL-ADMISSION-V1.md']) == RULE_SHA256, 'frozen admission rule changed')
    with gzip.GzipFile(fileobj=io.BytesIO(artifacts['selection-tables.json.gz']),mode='rb') as stream:
        raw = stream.read(16*1024*1024+1)
    require(len(raw) <= 16*1024*1024, 'decompressed tables exceed bound')
    require(sha(raw) == expected['recomputation']['tables_uncompressed_sha256'], 'decompressed table hash mismatch')
    tables, record = parse(raw), parse(artifacts['selection-result.json'])
    module = types.ModuleType('_closed_portable_selection')
    module.__file__ = 'causal_selection.py'
    exec(compile(artifacts['causal_selection.py'],module.__file__,'exec'),module.__dict__)
    require(module.RULE_DOCUMENT_SHA256 == RULE_SHA256, 'selector/rule binding differs')
    checked = module.verify_selection(tables,record)
    require(checked['sha256'] == expected['recomputation']['selection_internal_sha256'], 'complete selection hash differs')
    require(checked['result']['selection_status'] == 'no-control-match'
        and checked['result']['target_unit'] is checked['result']['control_unit'] is None,
        'historical no-control-match changed')
    return {'outcome':'selection-reproduced','selection_status':checked['result']['selection_status'],
        'selection_internal_sha256':checked['sha256'],'declared_cardinality':checked['result']['declared_cardinality'],
        'scope':'complete reduced-table aggregation and unchanged selection; not raw capture or neural replay',
        'model_forwards':0,'native_queries':0,'raw_capture_bytes_reverified':False,'scientific_admission':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir),sort_keys=True))


if __name__ == '__main__':
    main()
