//! hytorch-seam: spool consumer, protocol v2. Spec v2.2 §06–07.
//!
//! Filesystem spool protocol (ring buffer comes with the fused GPU seam):
//!
//!   spool/run-start.json                ← trainer writes ONCE before step 0:
//!       {"manifest_sha256":…, "build.apply_ref_hash":…, "build.harness_commit":…,
//!        "build.torch_wheel":…, "build.backend_wheel":…, "infra": {...},
//!        "policy": {"policy_id":…, "k":…, "s_slots":…, "n_features":…, "mag_max":…}}
//!   spool/run-start.ack                 → seam wrote RUN_START+POLICY into Hyphae
//!
//!   spool/step-<S>-layer-<L>.frame      ← per layer (elided wire)
//!   spool/step-<S>.barrier              ← trainer: "all frames spooled"
//!   spool/step-<S>.receipt              → seam: final_head hex (gates opt.step)
//!
//!   spool/step-<S>.facts                ← trainer AFTER opt.step()+renorm:
//!       line 1: lr grad_norm policy_id
//!       line 2: c_prev_hex   line 3: c_next_hex
//!       line 4: resets (comma-separated feature ids, may be empty)
//!       line 5: seal_theta_hex or "-"
//!   spool/step-<S>.chainack             → seam committed the STEP chain
//!
//!   spool/export.json                   ← trainer at the end
//!   spool/export.ack                    → EXPORT fact committed
//!
//! Fail-stop: on ANY commit error the seam writes NO ack; the trainer's
//! barrier times out and the run dies (spec §5.4).

use seam::{frames::decode_frame, SeamStore, StepFacts, StepFrames};
use std::path::{Path, PathBuf};
use std::time::Duration;

fn hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

type Facts = (f64, f64, u64, [u8; 32], [u8; 32], Vec<u32>, Option<[u8; 32]>, Vec<(String, Vec<f64>)>);

fn parse_facts(path: &Path) -> Option<Facts> {
    let txt = std::fs::read_to_string(path).ok()?;
    let mut lines = txt.lines();
    let mut first = lines.next()?.split_whitespace();
    let lr: f64 = first.next()?.parse().ok()?;
    let grad_norm: f64 = first.next()?.parse().ok()?;
    let policy_id: u64 = first.next()?.parse().ok()?;
    let c_prev = hex32(lines.next()?)?;
    let c_next = hex32(lines.next()?)?;
    let resets_line = lines.next().unwrap_or("").trim();
    let resets: Vec<u32> = if resets_line.is_empty() {
        vec![]
    } else {
        resets_line.split(',').filter_map(|t| t.trim().parse().ok()).collect()
    };
    let seal_line = lines.next().unwrap_or("-").trim();
    let seal = if seal_line == "-" { None } else { hex32(seal_line) };
    // line 6 (optional): Law-0 bypass scalars "name=v,v,...;name2=v,..."
    let bypass_line = lines.next().unwrap_or("").trim();
    let bypass: Vec<(String, Vec<f64>)> = if bypass_line.is_empty() || bypass_line == "-" {
        vec![]
    } else {
        bypass_line
            .split(';')
            .filter_map(|grp| {
                let (name, vals) = grp.split_once('=')?;
                let v: Vec<f64> = vals.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                Some((name.trim().to_string(), v))
            })
            .collect()
    };
    Some((lr, grad_norm, policy_id, c_prev, c_next, resets, seal, bypass))
}

/// Minimal JSON string-field extractor (no serde dep for the seam bin).
fn jstr(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = json.find(&pat)? + pat.len();
    let rest = &json[i..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let rest = &rest[1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn jnum(json: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = json.find(&pat)? + pat.len();
    let rest = &json[i..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn write_atomic(path: &Path, contents: &str) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, contents).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: hytorch-seam <data-dir> <run-id> <spool-dir> [--once]");
        std::process::exit(2);
    }
    let (data_dir, run_id, spool) = (&args[1], &args[2], PathBuf::from(&args[3]));
    let once = args.iter().any(|a| a == "--once");

    let mut store = SeamStore::open(data_dir, run_id).expect("open hyphae");
    // Wire retention cadence: raw facts persisted every N steps (1 = all,
    // the default). The T2 chain + counts are ALWAYS durable for every step;
    // only the raw wire bytes are sampled. Declared in RUN_START via env.
    let wire_every: u64 = std::env::var("HYTORCH_WIRE_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);
    eprintln!(
        "[seam] open data={data_dir} run={run_id} spool={} wire_every={wire_every}",
        spool.display()
    );

    // Refusals are terminal per content: refuse ONCE, remember, stay silent.
    // (A 1ms retry loop printing refusals forever fills the stderr pipe and
    // freezes the seam — found by the E2E refusal test.)
    let mut refused_run_start: Option<String> = None;
    let mut refused_steps: std::collections::HashSet<u64> = Default::default();

    loop {
        let mut worked = false;

        // ---- RUN_START handshake ----
        let rs_path = spool.join("run-start.json");
        let rs_ack = spool.join("run-start.ack");
        if rs_path.exists() && !rs_ack.exists() {
            let json = std::fs::read_to_string(&rs_path).unwrap_or_default();
            if refused_run_start.as_deref() == Some(json.as_str()) {
                // Already refused this exact content; wait for a new file.
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let get = |k: &str| jstr(&json, k).unwrap_or_default();
            let mut infra: Vec<(&str, String)> = vec![
                ("device_slug", get("device_slug")),
                ("region", get("region")),
                ("procurement", get("procurement")),
                ("driver", get("driver")),
            ];
            // Child-run citation (§5.4 resume protocol): a preempted run is
            // continued by a NEW run_id whose RUN_START cites the parent run,
            // its last receipt head, and the checkpoint step resumed from.
            // Empty when this is a root run.
            for k in ["parent_run", "parent_head", "resume_step"] {
                let v = get(k);
                if !v.is_empty() {
                    infra.push((k, v));
                }
            }
            let started = store.run_start(
                &get("manifest_sha256"),
                &get("build.apply_ref_hash"),
                &get("build.harness_commit"),
                &get("build.torch_wheel"),
                &get("build.backend_wheel"),
                infra,
            );
            match started {
                Ok(()) => {
                    let selection =
                        jstr(&json, "selection").unwrap_or_else(|| "global_topk".into());
                    // Fail-stop (review 2026-09-02): a RUN_START missing policy
                    // fields used to be silently filled with toy defaults
                    // (policy 1, k 4, ...). The policy IS the contract; refuse.
                    let need = |k: &str| -> Result<f64, String> {
                        jnum(&json, k).ok_or_else(|| format!("RUN_START refused: policy field {k} missing"))
                    };
                    let pol_ok = match (need("policy_id"), need("k"), need("s_slots"), need("n_features"), need("mag_max")) {
                        (Ok(pid), Ok(k), Ok(ss), Ok(nf), Ok(mm)) => store.put_policy(
                            pid as u64, k as u16, ss as u8, nf as u16, mm, &selection,
                            jnum(&json, "proposal_clip").unwrap_or(0.0),
                        ),
                        (a, b, c, d, e) => Err([a.err(), b.err(), c.err(), d.err(), e.err()]
                            .into_iter().flatten().collect::<Vec<_>>().join("; ")),
                    };
                    match pol_ok {
                        Ok(()) => {
                            write_atomic(&rs_ack, "ok\n");
                            eprintln!("[seam] RUN_START + POLICY committed");
                            worked = true;
                        }
                        Err(e) => eprintln!("[seam] POLICY FAILED: {e}"),
                    }
                }
                Err(e) => {
                    eprintln!("[seam] RUN_START REFUSED: {e}");
                    refused_run_start = Some(json.clone());
                }
            }
        }

        // ---- Phase A: barriers ----
        let mut barrier_files: Vec<PathBuf> = std::fs::read_dir(&spool)
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "barrier"))
                    .collect()
            })
            .unwrap_or_default();
        barrier_files.sort();

        for bpath in barrier_files {
            let stem = bpath.file_stem().unwrap().to_string_lossy().to_string();
            let Some(step_id) = stem.strip_prefix("step-").and_then(|s| s.parse::<u64>().ok())
            else {
                continue;
            };
            if refused_steps.contains(&step_id) {
                continue;
            }
            let mut frame_paths: Vec<PathBuf> = std::fs::read_dir(&spool)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .map(|n| {
                            let n = n.to_string_lossy();
                            n.starts_with(&format!("step-{step_id}-layer-"))
                                && n.ends_with(".frame")
                        })
                        .unwrap_or(false)
                })
                .collect();
            frame_paths.sort();
            if frame_paths.is_empty() {
                continue;
            }

            let mut frames = StepFrames::default();
            let mut ok = true;
            for fp in &frame_paths {
                let bytes = std::fs::read(fp).unwrap_or_default();
                match decode_frame(&bytes) {
                    Ok((h, wire)) => frames.push(h, wire.to_vec()),
                    Err(e) => {
                        eprintln!("[seam] bad frame {}: {e:?}", fp.display());
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }

            let retain = step_id % wire_every == 0;
            match store.commit_layers(&frames, step_id, retain) {
                Ok(receipt) => {
                    let head_hex: String =
                        receipt.final_head.iter().map(|b| format!("{b:02x}")).collect();
                    write_atomic(
                        &spool.join(format!("step-{step_id}.receipt")),
                        &format!("{head_hex}\n"),
                    );
                    eprintln!(
                        "[seam] step {step_id}: {} layers, {} persisted, {} elided, head {head_hex}",
                        receipt.n_layers, receipt.n_persisted, receipt.n_elided
                    );
                    for fp in frame_paths {
                        let _ = std::fs::remove_file(fp);
                    }
                    let _ = std::fs::remove_file(&bpath);
                    worked = true;
                }
                Err(e) => {
                    if refused_steps.insert(step_id) {
                        eprintln!("[seam] LAYER COMMIT FAILED step {step_id}: {e}");
                    }
                    // no receipt → trainer fail-stops (log once, stay silent)
                }
            }
        }

        // ---- Phase B: step chains ----
        let mut facts_files: Vec<PathBuf> = std::fs::read_dir(&spool)
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "facts"))
                    .collect()
            })
            .unwrap_or_default();
        facts_files.sort();

        for facts_path in facts_files {
            let stem = facts_path.file_stem().unwrap().to_string_lossy().to_string();
            let Some(step_id) = stem.strip_prefix("step-").and_then(|s| s.parse::<u64>().ok())
            else {
                continue;
            };
            if refused_steps.contains(&step_id) {
                continue;
            }
            let Some((lr, grad_norm, policy_id, c_prev, c_next, resets, seal, bypass)) =
                parse_facts(&facts_path)
            else {
                eprintln!("[seam] bad facts {}", facts_path.display());
                continue;
            };
            let facts = StepFacts {
                step_id, lr, grad_norm, c_prev, c_next, policy_id,
                resets, seal_theta: seal, bypass,
            };
            match store.commit_step_chain(&facts) {
                Ok(()) => {
                    write_atomic(&spool.join(format!("step-{step_id}.chainack")), "ok\n");
                    let _ = std::fs::remove_file(&facts_path);
                    worked = true;
                }
                Err(e) => {
                    if refused_steps.insert(step_id) {
                        eprintln!("[seam] STEP CHAIN FAILED step {step_id}: {e}");
                    }
                    // no chainack → trainer fail-stops at next barrier.
                    // Refusal is terminal: log once, stay silent (a 1ms loop
                    // spamming stderr fills the pipe and freezes the seam).
                }
            }
        }

        // ---- GENFACT: journaled inference (runtime.py) ----
        // Files: genfact-NNNNNNNN.json = {"key": "gen/...", "value": {...}}
        // Commit into Hyphae under run/<id>/<key>, ack, consume. The chat
        // runtime treats a missing ack as a VISIBLE citability degradation,
        // never a silent one.
        let mut gen_files: Vec<PathBuf> = std::fs::read_dir(&spool)
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| {
                        p.file_name()
                            .map(|n| {
                                let n = n.to_string_lossy();
                                n.starts_with("genfact-") && n.ends_with(".json")
                            })
                            .unwrap_or(false)
                    })
                    .collect()
            })
            .unwrap_or_default();
        gen_files.sort();
        for gf in gen_files {
            let json = std::fs::read_to_string(&gf).unwrap_or_default();
            let key = jstr(&json, "key").unwrap_or_default();
            if key.is_empty() {
                let _ = std::fs::remove_file(&gf);
                continue;
            }
            // Store the raw JSON payload as a string fact (queryable, hash-
            // chained by the runtime's own head field inside the payload).
            let value_start = json.find("\"value\"").and_then(|i| json[i..].find(':').map(|j| i + j + 1));
            let payload = value_start
                .map(|i| json[i..].trim().trim_end_matches('}').to_string())
                .unwrap_or_default();
            match store.put_fact(
                &key,
                vec![
                    ("kind", hyphae_query::Value::String("GENFACT".into())),
                    ("payload", hyphae_query::Value::String(payload)),
                ],
            ) {
                Ok(()) => {
                    let ack = gf.with_extension("ack");
                    write_atomic(&ack, "ok\n");
                    let _ = std::fs::remove_file(&gf);
                    worked = true;
                }
                Err(e) => {
                    eprintln!("[seam] GENFACT FAILED {key}: {e}");
                    let _ = std::fs::remove_file(&gf); // do not spin on a bad fact
                }
            }
        }

        // ---- EXPORT ----
        let ex_path = spool.join("export.json");
        let ex_ack = spool.join("export.ack");
        if ex_path.exists() && !ex_ack.exists() {
            let json = std::fs::read_to_string(&ex_path).unwrap_or_default();
            let step_id = jnum(&json, "step_id").unwrap_or(0.0) as u64;
            match store.put_export(
                step_id,
                &jstr(&json, "theta_sha256").unwrap_or_default(),
                &jstr(&json, "c_sha256").unwrap_or_default(),
                &jstr(&json, "final_head").unwrap_or_default(),
            ) {
                Ok(()) => {
                    write_atomic(&ex_ack, "ok\n");
                    eprintln!("[seam] EXPORT committed");
                    worked = true;
                }
                Err(e) => eprintln!("[seam] EXPORT FAILED: {e}"),
            }
        }

        if once && !worked {
            break;
        }
        if !worked {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
