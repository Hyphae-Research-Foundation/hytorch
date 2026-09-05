//! hytorch-ledger-query — read-only lookups against a run's ledger.
//!
//! The resume protocol (§5.4) needs the LAST receipt head of a preempted
//! run to cite it in the child's RUN_START; custody needs arbitrary record
//! dumps without hand-rolled one-off crates (as done manually after 3.4).
//!
//! Usage:
//!   hytorch-ledger-query <data-dir> last-head <run-id> [max-step]
//!       → "<step> <head-hex>" of the highest RECEIPT at or below max-step
//!         (scans down from max-step; default 1_000_000).
//!   hytorch-ledger-query <data-dir> get <full-key>
//!       → debug-print of one record's value.

use hyphae_engine::HyphaeEngine;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hytorch-ledger-query <data-dir> <last-head|get> <args…>");
        std::process::exit(2);
    }
    let opened = HyphaeEngine::open(&args[1]).expect("open hyphae");
    let engine = opened.engine;

    match args[2].as_str() {
        "last-head" => {
            let run_id = &args[3];
            let max_step: u64 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
            for step in (0..=max_step).rev() {
                let key = format!("run/{run_id}/step/{step:08}/RECEIPT");
                if let Ok(Some(rec)) = engine.get_record(key.as_bytes()) {
                    let dbg = format!("{:?}", rec.value);
                    // final_head is a 64-hex string in the record map.
                    if let Some(i) = dbg.find("final_head") {
                        let rest = &dbg[i..];
                        if let Some(h) = rest
                            .split('"')
                            .find(|s| s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit()))
                        {
                            println!("{step} {h}");
                            return;
                        }
                    }
                }
            }
            eprintln!("no RECEIPT found for run {run_id}");
            std::process::exit(1);
        }
        "get" => {
            let key = &args[3];
            match engine.get_record(key.as_bytes()) {
                Ok(Some(rec)) => println!("{:?}", rec.value),
                Ok(None) => {
                    eprintln!("not found: {key}");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("error: {e:?}");
                    std::process::exit(1);
                }
            }
        }
        // dump-wire <run-id> <step> <out-dir> [max-frames]
        // Writes the raw retained wire of one WIRED step as frame-NNNNN.bin
        // files (concatenable; 16B records). Layer id rides in the wire
        // records themselves, so files need no metadata.
        "dump-wire" => {
            let run_id = &args[3];
            let step: u64 = args[4].parse().expect("step");
            let out_dir = std::path::PathBuf::from(&args[5]);
            let max_frames: usize =
                args.get(6).and_then(|v| v.parse().ok()).unwrap_or(200_000);
            std::fs::create_dir_all(&out_dir).unwrap();
            let mut n_frames = 0usize;
            let mut n_bytes = 0usize;
            'outer: for frame in 0..max_frames {
                // layer for this frame is unknown; probe the full unit range
                // (in-process gets are cheap; misses are cheaper).
                let mut hit = false;
                for layer in 0..4096u32 {
                    let key = format!(
                        "run/{run_id}/step/{step:08}/frame/{frame:05}/layer/{layer:04}"
                    );
                    if let Ok(Some(rec)) = engine.get_record(key.as_bytes()) {
                        hit = true;
                        if let hyphae_query::Value::Object(map) = &rec.value {
                            match map.get("wire") {
                                Some(hyphae_query::Value::Bytes(b)) => {
                                    std::fs::write(
                                        out_dir.join(format!("frame-{frame:05}.bin")),
                                        b,
                                    )
                                    .unwrap();
                                    n_frames += 1;
                                    n_bytes += b.len();
                                }
                                _ => {
                                    eprintln!(
                                        "frame {frame}: wire not retained (elided step?)"
                                    );
                                    break 'outer;
                                }
                            }
                        }
                        break;
                    }
                }
                if !hit {
                    break; // no more frames for this step
                }
            }
            println!("{n_frames} frames, {n_bytes} bytes");
        }
        other => {
            eprintln!("unknown subcommand {other}");
            std::process::exit(2);
        }
    }
}
