"""Pure selection over declared pretreatment tables, never scientific admission.

No model, Torch, RNG, file, runner, environment or provider operations. The frozen
rule is docs/FACTUAL-CAUSAL-ADMISSION-V1.md. Real competence, checkpoint, membership,
exposure and measurement provenance must be bound by a separate admission layer.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

TABLE_SCHEMA = "hytorch.causal-selection-tables.v1"
RESULT_SCHEMA = "hytorch.causal-selection-result.v1"
RULE_DOCUMENT_SHA256 = "a9f1aef9f00e5bb98aa7fe9e294771f279c0693a9815ae40a1ea34167d469f79"
POSITIONS = ("identity", "heldout-shift8", "heldout-gaps4")
UNITS = (0, 1, 2, 3)
MIN_DELTA = 0.05
NLL_DRIFT = 0.00001
RMS_DRIFT = 0.000001
LOSS_FLOOR = 0.02
LOSS_FRACTION = 0.20
RMS_FACTOR = 1.5
ADMISSION_CALIPER = 0.10


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def rule_contract():
    return {"document_sha256": RULE_DOCUMENT_SHA256, "units": list(UNITS), "positions": list(POSITIONS),
        "batteries": 2, "minimum_delta_value_nats": MIN_DELTA,
        "repeat_max_abs_nll_drift": NLL_DRIFT, "repeat_max_rel_rms_drift": RMS_DRIFT,
        "loss_scale_floor": LOSS_FLOOR, "loss_scale_target_fraction": LOSS_FRACTION,
        "update_rms_ratio_factor": RMS_FACTOR, "relative_update_rms_ratio_factor": RMS_FACTOR,
        "admission_fraction_caliper": ADMISSION_CALIPER, "tie_break": "lowest-unit-index",
        "positive_floor_comparison": ">", "caliper_comparison": "inclusive; no extra tolerance",
        "rms_repeat_reference": "first battery; zero/zero has zero drift; zero/nonzero is unstable"}


def _keys(value, keys, where):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{where}: exact fields required: {sorted(keys)}")


def _list(value, where, *, nonempty=True):
    if type(value) is not list or (nonempty and not value):
        raise ValueError(f"{where}: {'nonempty ' if nonempty else ''}list required")
    return value


def _id(value, where):
    if type(value) is not str or not value or "\0" in value:
        raise ValueError(f"{where}: nonempty string identity required")
    return value


def _integer(value, where, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: integer >= {minimum} required")
    return value


def _number(value, where):
    if type(value) not in (int, float):
        raise ValueError(f"{where}: finite nonnegative number required")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{where}: finite nonnegative number required") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{where}: finite nonnegative number required")
    return value


def _finite(value):
    if not math.isfinite(value):
        raise ValueError("derived arithmetic is nonfinite")
    return value


def _mean(values):
    values = list(values)
    return _finite(math.fsum(value / len(values) for value in values))


def _unit(value, *, healthy=False):
    if healthy and value is None:
        return None
    if type(value) is not int or value not in UNITS:
        raise ValueError("unit must be an integer U0..U3; only healthy NLL rows use null")
    return value


def _position(value):
    if type(value) is not str or value not in POSITIONS:
        raise ValueError("position must be one of the three frozen conditions")
    return value


def _put(mapping, key, value, where):
    if key in mapping:
        raise ValueError(f"duplicate {where} coordinate")
    mapping[key] = value


def _membership(mapping, expected, where):
    if set(mapping) != expected:
        raise ValueError(f"{where}: complete Cartesian membership required; missing={len(expected-set(mapping))}, extra={len(set(mapping)-expected)}")


def _validate(tables):
    _keys(tables, {"schema", "rule_document_sha256", "scope", "n_units", "positions", "residual_width",
                   "candidate_k", "facts", "control_records", "batteries"}, "tables")
    if (tables["schema"] != TABLE_SCHEMA or tables["rule_document_sha256"] != RULE_DOCUMENT_SHA256
            or tables["scope"] not in ("fixture", "unadmitted-measurements")):
        raise ValueError("table schema, frozen rule or non-admitted scope differs")
    if type(tables["n_units"]) is not int or tables["n_units"] != 4 or tables["positions"] != list(POSITIONS):
        raise ValueError("exactly four units and all three ordered position conditions required")
    width = _integer(tables["residual_width"], "residual width", minimum=1)
    k = _integer(tables["candidate_k"], "candidate k", minimum=1)
    facts, prompts, controls = {}, set(), {}
    for fact in _list(tables["facts"], "facts"):
        _keys(fact, {"fact_id", "analysis_split", "exposure", "realized_exposures", "prompt_ids"}, "fact")
        fid = _id(fact["fact_id"], "fact ID")
        if fact["analysis_split"] != "discovery" or fact["exposure"] != "common":
            raise ValueError("localizer inventory must declare discovery common facts only")
        _integer(fact["realized_exposures"], "realized exposures", minimum=1)
        ids = [_id(value, "prompt ID") for value in _list(fact["prompt_ids"], "prompt IDs")]
        if len(set(ids)) != len(ids) or prompts.intersection(ids):
            raise ValueError("duplicate prompt identity")
        prompts.update(ids)
        _put(facts, fid, tuple(ids), "fact ID")
    for row in _list(tables["control_records"], "control records"):
        _keys(row, {"record_id", "analysis_split", "valid_targets"}, "control record")
        if row["analysis_split"] != "discovery":
            raise ValueError("control text inventory must declare discovery only")
        _put(controls, _id(row["record_id"], "control record ID"),
             _integer(row["valid_targets"], "valid targets", minimum=1), "control record ID")
    batteries = _list(tables["batteries"], "batteries")
    if len(batteries) != 2:
        raise ValueError("exactly two batteries required")
    expected_local = {(fid, pid, pos, unit) for fid, pids in facts.items() for pid in pids
                      for pos in POSITIONS for unit in (None, *UNITS)}
    expected_general = {(rid, pos, unit) for rid in controls for pos in POSITIONS for unit in (None, *UNITS)}
    expected_writes = {(rid, pos, unit) for rid in controls for pos in POSITIONS for unit in UNITS}
    parsed = []
    for repetition, battery in enumerate(batteries):
        _keys(battery, {"repetition", "localization", "general_nll", "healthy_writes"}, "battery")
        if type(battery["repetition"]) is not int or battery["repetition"] != repetition:
            raise ValueError("battery repetition must be ordered 0,1")
        local, general, writes = {}, {}, {}
        for row in _list(battery["localization"], "localization"):
            _keys(row, {"fact_id", "prompt_id", "position", "unit", "value_nll"}, "localization row")
            key = (_id(row["fact_id"], "fact ID"), _id(row["prompt_id"], "prompt ID"),
                   _position(row["position"]), _unit(row["unit"], healthy=True))
            _put(local, key, _number(row["value_nll"], "Value NLL"), "localization")
        for row in _list(battery["general_nll"], "general NLL"):
            _keys(row, {"record_id", "position", "unit", "nll_sum", "valid_targets"}, "general NLL row")
            rid = _id(row["record_id"], "control record ID")
            if rid not in controls or _integer(row["valid_targets"], "row valid targets", minimum=1) != controls[rid]:
                raise ValueError("general NLL target count differs from the declared record")
            key = rid, _position(row["position"]), _unit(row["unit"], healthy=True)
            _put(general, key, _number(row["nll_sum"], "general NLL sum"), "general NLL")
        for row in _list(battery["healthy_writes"], "healthy writes"):
            _keys(row, {"record_id", "position", "unit", "applied_sum_squares", "residual_sum_squares",
                        "element_count", "admitted", "overflow", "original_abort", "total_candidates"}, "healthy write row")
            rid = _id(row["record_id"], "control record ID")
            if rid not in controls:
                raise ValueError("write record absent from declared control inventory")
            key = rid, _position(row["position"]), _unit(row["unit"])
            values = {name: _integer(row[name], name) for name in
                      ("element_count", "admitted", "overflow", "original_abort", "total_candidates")}
            if (values["element_count"] != controls[rid]*width or values["total_candidates"] != controls[rid]*k
                    or sum(values[name] for name in ("admitted", "overflow", "original_abort")) != values["total_candidates"]):
                raise ValueError("write element/candidate counts do not match declared denominators")
            values.update({name: _number(row[name], name) for name in ("applied_sum_squares", "residual_sum_squares")})
            if values["admitted"] == 0 and values["applied_sum_squares"] != 0:
                raise ValueError("zero admitted writes cannot have a nonzero applied contribution")
            _put(writes, key, values, "healthy write")
        _membership(local, expected_local, "localization")
        _membership(general, expected_general, "general NLL")
        _membership(writes, expected_writes, "healthy writes")
        parsed.append({"local": local, "general": general, "writes": writes})
    return facts, controls, parsed


def _summaries(facts, controls, battery):
    local, general, writes = battery["local"], battery["general"], battery["writes"]
    results = []
    valid_targets = sum(controls.values())
    for unit in UNITS:
        fact_effects = [{"fact_id": fid, "delta_value_nats": _mean(
            local[fid, pid, pos, unit]-local[fid, pid, pos, None] for pid in pids for pos in POSITIONS)}
            for fid, pids in sorted(facts.items())]
        by_position = []
        for pos in POSITIONS:
            rows = [writes[rid, pos, unit] for rid in sorted(controls)]
            totals = {name: sum(row[name] for row in rows) for name in
                      ("element_count", "admitted", "overflow", "original_abort", "total_candidates")}
            totals.update({name: _finite(math.fsum(row[name] for row in rows)) for name in
                           ("applied_sum_squares", "residual_sum_squares")})
            totals.update(position=pos, valid_targets=valid_targets,
                general_nll_healthy=_finite(math.fsum(general[rid, pos, None] for rid in sorted(controls))/valid_targets),
                general_nll_veto=_finite(math.fsum(general[rid, pos, unit] for rid in sorted(controls))/valid_targets),
                delta_value_nats=_mean(_mean(local[fid, pid, pos, unit]-local[fid, pid, pos, None]
                                            for pid in pids) for fid, pids in sorted(facts.items())))
            by_position.append(totals)
        totals = {name: sum(row[name] for row in by_position) for name in
                  ("element_count", "admitted", "overflow", "original_abort", "total_candidates")}
        totals.update({name: _finite(math.fsum(row[name] for row in by_position)) for name in
                       ("applied_sum_squares", "residual_sum_squares")})
        m = _finite(math.sqrt(totals["applied_sum_squares"]/totals["element_count"]))
        r = _finite(math.sqrt(totals["residual_sum_squares"]/totals["element_count"]))
        results.append({"unit": unit, "delta_value_nats": _mean(row["delta_value_nats"] for row in fact_effects),
            "fact_effects": fact_effects, "positions": by_position, "totals": totals,
            "g": _mean(row["general_nll_veto"]-row["general_nll_healthy"] for row in by_position),
            "m": m, "r": r, "rho": _finite(m/r) if r else None,
            "p": _finite(totals["admitted"]/totals["total_candidates"])})
    return results


def _match(units, target):
    t = units[target]
    scale = max(LOSS_FLOOR, LOSS_FRACTION*abs(t["g"]))
    candidates = []
    for unit in UNITS:
        if unit == target:
            continue
        u = units[unit]
        invalid = [f"{side}.{name}=zero" for side, value in (("target", t), ("control", u))
                   for name in ("m", "r", "p") if value[name] == 0]
        if invalid:
            candidates.append({"unit": unit, "admissible": False, "undefined_distance_reasons": invalid,
                               "calipers": None, "distance": None})
            continue
        gdiff = _finite(u["g"]-t["g"])
        mr = _finite(u["m"]/t["m"])
        rr = _finite(u["rho"]/t["rho"])
        pdiff = _finite(u["p"]-t["p"])
        checks = {"general_nll": abs(gdiff) <= scale, "update_rms": 1/RMS_FACTOR <= mr <= RMS_FACTOR,
                  "relative_update_rms": 1/RMS_FACTOR <= rr <= RMS_FACTOR,
                  "admission_fraction": abs(pdiff) <= ADMISSION_CALIPER}
        if mr == 0 or rr == 0:
            raise ValueError("matching ratio underflow is not a usable finite measurement")
        distance = _finite((gdiff/scale)**2 + (math.log(mr)/math.log(RMS_FACTOR))**2
                           + (math.log(rr)/math.log(RMS_FACTOR))**2 + (pdiff/ADMISSION_CALIPER)**2)
        candidates.append({"unit": unit, "admissible": all(checks.values()), "calipers": checks,
            "distance": distance, "g_difference": gdiff, "update_rms_ratio": mr,
            "relative_update_rms_ratio": rr, "admission_fraction_difference": pdiff})
    admissible = [row for row in candidates if row["admissible"]]
    chosen = min(admissible, key=lambda row:(row["distance"], row["unit"]))["unit"] if admissible else None
    return {"target_unit": target, "general_loss_scale": scale, "candidates": candidates,
            "admissible_units": [row["unit"] for row in admissible], "control_unit": chosen}


def _drift(first, second):
    if first == second:
        return 0.0
    if first == 0:
        return None
    return _finite(abs(second-first)/abs(first))


def _raw_repeat_checks(first, second):
    """Check corresponding healthy measurements before aggregates can cancel."""
    failures, classifications = [], []
    maximum, undefined = 0.0, 0
    for key in sorted(first):
        a, b = first[key], second[key]
        coordinate = {"record_id": key[0], "position": key[1], "unit": key[2]}
        for name, field in (("m", "applied_sum_squares"), ("r", "residual_sum_squares")):
            left = _finite(math.sqrt(a[field]/a["element_count"]))
            right = _finite(math.sqrt(b[field]/b["element_count"]))
            drift = _drift(left, right)
            if drift is None:
                undefined += 1
            else:
                maximum = max(maximum, drift)
            if drift is None or drift > RMS_DRIFT:
                failures.append({**coordinate, "measurement": name, "first_rms": left,
                                 "second_rms": right, "relative_drift": drift})
        for field in ("admitted", "overflow", "original_abort"):
            if a[field] != b[field]:
                classifications.append({**coordinate, "classification": field,
                                        "first": a[field], "second": b[field]})
    return {"rms_measurements_checked": 2*len(first), "max_defined_relative_rms_drift": maximum,
            "zero_to_nonzero_rms_measurements": undefined, "rms_failures": failures,
            "candidate_classification_differences": classifications,
            "measurement_integrity_stable": not failures and not classifications}


def select_units(tables):
    """Validate every declared row, then apply the frozen two-battery rule.

    Positive selection is not competence or provenance admission. Input objects
    remain unchanged. All three control candidates and both batteries are kept.
    """
    try:
        facts, controls, parsed = _validate(tables)
        input_sha = _sha(tables)
        summaries = [_summaries(facts, controls, battery) for battery in parsed]
        rankings = [sorted(UNITS, key=lambda unit:(-rows[unit]["delta_value_nats"], unit)) for rows in summaries]
        targets = [ranking[0] for ranking in rankings]
        local_drift = max(abs(parsed[1]["local"][key]-value) for key, value in parsed[0]["local"].items())
        general_drift = max(abs(parsed[1]["general"][key]-value)/controls[key[0]]
                            for key, value in parsed[0]["general"].items())
        rms_drifts = [{"unit": unit, "measurement": name,
                       "relative_drift": _drift(summaries[0][unit][name], summaries[1][unit][name])}
                      for unit in UNITS for name in ("m", "r")]
        counts_equal = all(row["admitted"] == parsed[1]["writes"][key]["admitted"]
                           for key, row in parsed[0]["writes"].items())
        raw_repeat = _raw_repeat_checks(parsed[0]["writes"], parsed[1]["writes"])
        positive = [summaries[rep][target]["delta_value_nats"] > MIN_DELTA for rep, target in enumerate(targets)]
        matching = None
        if targets[0] != targets[1] or local_drift > NLL_DRIFT:
            status = "unstable-localizer"
        elif not any(positive):
            status = "no-positive-target"
        elif not all(positive):
            status = "unstable-localizer"
        else:
            matching = [_match(rows, targets[0]) for rows in summaries]
            stable = (general_drift <= NLL_DRIFT and counts_equal and raw_repeat["measurement_integrity_stable"]
                      and all(row["relative_drift"] is not None and row["relative_drift"] <= RMS_DRIFT for row in rms_drifts)
                      and matching[0]["admissible_units"] == matching[1]["admissible_units"]
                      and matching[0]["control_unit"] == matching[1]["control_unit"])
            if not stable:
                status = "unstable-matching"
            elif matching[0]["control_unit"] is None:
                status = "no-control-match"
            else:
                status = "selected"
        result = {"schema": RESULT_SCHEMA, "scope": tables["scope"], "scientific_admission": False,
            "pending_real_bindings": ["competence-report", "checkpoint-and-exposure-provenance",
                "membership-against-frozen-real-inventories", "measurement-and-hardware-qualification"],
            "input_sha256": input_sha, "rule": rule_contract(), "rule_sha256": _sha(rule_contract()),
            "selection_status": status, "target_unit": targets[0] if status == "selected" else None,
            "control_unit": matching[0]["control_unit"] if status == "selected" else None,
            "tentative_targets": targets, "positive_floor_passes": positive, "batteries": summaries,
            "localizer_rankings": [{"units": ranking, "runner_up_gap_nats":
                _finite(summaries[rep][ranking[0]]["delta_value_nats"]-summaries[rep][ranking[1]]["delta_value_nats"])}
                for rep, ranking in enumerate(rankings)],
            "matching": matching, "repeat_checks": {"max_abs_localizer_nll_drift": local_drift,
                "max_abs_control_record_nll_drift": general_drift, "rms_drifts": rms_drifts,
                "all_integer_admission_counts_identical": counts_equal, "raw_measurements": raw_repeat},
            "declared_cardinality": {"facts": len(facts), "prompts": sum(map(len, facts.values())),
                "control_records": len(controls), "batteries": 2, "units": 4, "positions": 3,
                "localizer_rows_per_battery": len(parsed[0]["local"]),
                "general_nll_rows_per_battery": len(parsed[0]["general"]),
                "healthy_write_rows_per_battery": len(parsed[0]["writes"])}}
        return {"result": result, "sha256": _sha(result)}
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("derived selection arithmetic overflowed or divided by zero") from error


def verify_selection(tables, record):
    """Recompute selection; a rehashed edited result cannot bypass the rule."""
    _keys(record, {"result", "sha256"}, "selection record")
    if (type(record["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            or record["sha256"] != _sha(record["result"])):
        raise ValueError("selection record fingerprint differs")
    expected = select_units(tables)
    if _canonical(record) != _canonical(expected):
        raise ValueError("selection record differs from complete recomputation")
    return deepcopy(expected)


def serialize_selection(tables, record):
    return _canonical(verify_selection(tables, record)).decode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def deserialize_selection(tables, serialized):
    return verify_selection(tables, json.loads(serialized, object_pairs_hook=_unique_object))
