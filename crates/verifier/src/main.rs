//! hytorch-verify: CLI. Reads a spill file, runs T1 (which includes the T2
//! recompute over the audited microbatch), prints verdict, exit code 0/1/2.
//!
//! Usage: hytorch-verify <spill-file> [--json]

use std::process::ExitCode;
use verifier::{read_spill, t1_replay, T1Outcome};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: hytorch-verify <spill-file> [--json]");
        return ExitCode::from(2);
    };
    let json = args.iter().any(|a| a == "--json");

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let spill = match read_spill(&bytes) {
        Ok(s) => s,
        Err(e) => {
            if json {
                println!("{{\"outcome\":\"spill_error\",\"detail\":\"{e:?}\"}}");
            } else {
                eprintln!("SPILL ERROR: {e:?} — the file itself is inconsistent");
            }
            return ExitCode::from(1);
        }
    };

    match t1_replay(&spill) {
        T1Outcome::Consistent { n_applied, final_head } => {
            let head_hex: String = final_head.iter().map(|b| format!("{b:02x}")).collect();
            if json {
                println!(
                    "{{\"outcome\":\"consistent\",\"step\":{},\"microbatch\":{},\"n_applied\":{},\"final_head\":\"{}\"}}",
                    spill.step_id, spill.microbatch_id, n_applied, head_hex
                );
            } else {
                println!(
                    "T1 CONSISTENT step={} mb={} applied={} head={}",
                    spill.step_id, spill.microbatch_id, n_applied, head_hex
                );
            }
            ExitCode::SUCCESS
        }
        T1Outcome::HeadMismatch { first_bad_layer, got, want_final } => {
            let g: String = got.iter().map(|b| format!("{b:02x}")).collect();
            let w: String = want_final.iter().map(|b| format!("{b:02x}")).collect();
            if json {
                println!(
                    "{{\"outcome\":\"head_mismatch\",\"layer\":{first_bad_layer},\"got\":\"{g}\",\"want\":\"{w}\"}}"
                );
            } else {
                eprintln!("T1 FAIL: head mismatch (≤ layer {first_bad_layer})\n  got  {g}\n  want {w}\nKILL THE RUN.");
            }
            ExitCode::from(1)
        }
        T1Outcome::ResidualMismatch { got, want } => {
            let g: String = got.iter().map(|b| format!("{b:02x}")).collect();
            let w: String = want.iter().map(|b| format!("{b:02x}")).collect();
            if json {
                println!(
                    "{{\"outcome\":\"residual_mismatch\",\"got\":\"{g}\",\"want\":\"{w}\"}}"
                );
            } else {
                eprintln!("T1 FAIL: chain intact but replayed residual differs (apply/journal/codebook do not close)\n  got  {g}\n  want {w}\nKILL THE RUN.");
            }
            ExitCode::from(1)
        }
        T1Outcome::ApplyError { layer, detail } => {
            if json {
                println!("{{\"outcome\":\"apply_error\",\"layer\":{layer},\"detail\":\"{detail}\"}}");
            } else {
                eprintln!("T1 FAIL: journal rejected by apply_ref at layer {layer}: {detail}\nKILL THE RUN.");
            }
            ExitCode::from(1)
        }
    }
}
