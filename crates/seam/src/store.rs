//! SeamStore: embedded Hyphae as the durable authority. Spec v2.2 §06–07.
//!
//! Two atomic batches per step, mirroring the spec loop exactly:
//!   Phase A (barrier): LAYER facts + RECEIPT — the receipt gates opt.step().
//!   Phase B (chain):   STEP record with REAL c_prev/c_next (post-opt.step,
//!                      post-renorm), plus CODEBOOK_RESET / SEAL when due.
//!
//! The store enforces codebook chain continuity: c_prev(t) must equal
//! c_next(t-1). A break means the trainer mutated C outside the ledger —
//! fail-stop, no chainack, the run dies (spec §4.2: EMA silencioso prohibido).

use binding_wire::{layer_head, LayerMeta, GENESIS_HEAD};
use hyphae_engine::HyphaeEngine;
use hyphae_query::{Record, Value};
use std::collections::BTreeMap;
use std::path::Path;
use uuid::Uuid;

use crate::frames::StepFrames;

pub struct SeamStore {
    engine: HyphaeEngine,
    run_id: String,
    /// c_next of the last committed STEP (continuity check).
    last_c_next: Option<[u8; 32]>,
    /// step_id of the last receipt issued (chain must reference it).
    last_receipt_step: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct StepFacts {
    pub step_id: u64,
    pub lr: f64,
    pub grad_norm: f64,
    pub c_prev: [u8; 32],
    pub c_next: [u8; 32],
    pub policy_id: u64,
    /// Features reset this step (CODEBOOK_RESET fact if non-empty).
    pub resets: Vec<u32>,
    /// H_canónico(θ) when the SEAL cadence fires.
    pub seal_theta: Option<[u8; 32]>,
    /// Law-0 bypass declaration (review 2026-09-02): residual mutations that
    /// happen OUTSIDE the catalog write (nanochat's per-layer
    /// `x = resid_lambda*x + x0_lambda*x0`). They are not silenced and not
    /// pretended away: the trainer reports the scalars every step and the
    /// ledger records them as a BYPASS fact next to the STEP. Empty when the
    /// architecture has no such path (phase 1/2 toy).
    pub bypass: Vec<(String, Vec<f64>)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepReceipt {
    pub step_id: u64,
    pub final_head: [u8; 32],
    pub n_layers: u16,
    pub n_persisted: u64,
    pub n_elided: u64,
}

fn unhex32(h: &str) -> Option<[u8; 32]> {
    if h.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn obj(fields: Vec<(&str, Value)>) -> Value {
    let mut m = BTreeMap::new();
    for (k, v) in fields {
        m.insert(k.to_string(), v);
    }
    Value::Object(m)
}

impl SeamStore {
    pub fn open(data_dir: impl AsRef<Path>, run_id: &str) -> Result<Self, String> {
        let opened = HyphaeEngine::open(data_dir).map_err(|e| format!("{e:?}"))?;
        let mut s = Self {
            engine: opened.engine,
            run_id: run_id.to_string(),
            last_c_next: None,
            last_receipt_step: None,
        };
        // Review 2026-09-02: continuity used to live only in memory, so the
        // c_prev(t) == c_next(t-1) check was silently skipped on the first
        // step after a seam restart — exactly the post-preemption resume
        // case. Rehydrate from the ledger: find the highest STEP record for
        // this run (bounded downward scan from the RECEIPT high-water mark)
        // and restore both watermarks.
        if let Some((step, c_next)) = s.last_step_from_ledger() {
            s.last_c_next = Some(c_next);
            // A receipt without its STEP chain (crash between phases) means
            // the trainer will retry that step's chain; leave the receipt
            // watermark at the STEP's step so commit_step_chain accepts it.
            s.last_receipt_step = Some(step);
        }
        Ok(s)
    }

    /// Highest (step_id, c_next) with a STEP record, scanning down from the
    /// highest RECEIPT. Cheap: receipts are dense, so the first hit is near
    /// the top; bounded by 1_000_000 steps.
    fn last_step_from_ledger(&self) -> Option<(u64, [u8; 32])> {
        // exponential probe up, then binary search down for the last RECEIPT
        let has = |st: u64| -> bool {
            self.engine
                .get_record(format!("run/{}/step/{:08}/RECEIPT", self.run_id, st).as_bytes())
                .ok()
                .flatten()
                .is_some()
        };
        if !has(0) {
            return None;
        }
        let mut lo = 0u64;
        let mut step = 1u64;
        while step < 1_000_000 && has(step) {
            lo = step;
            step *= 2;
        }
        let mut hi_b = step.min(1_000_000);
        while lo + 1 < hi_b {
            let mid = (lo + hi_b) / 2;
            if has(mid) { lo = mid } else { hi_b = mid }
        }
        let top = lo;
        // walk down (a few steps at most) to the last STEP record
        let mut st = top as i64;
        while st >= 0 {
            let key = format!("run/{}/step/{:08}/STEP", self.run_id, st);
            if let Ok(Some(rec)) = self.engine.get_record(key.as_bytes()) {
                if let Value::Object(map) = &rec.value {
                    if let Some(Value::String(h)) = map.get("c_next") {
                        if let Some(bytes) = unhex32(h) {
                            return Some((st as u64, bytes));
                        }
                    }
                }
            }
            st -= 1;
            if top as i64 - st > 64 {
                break;
            }
        }
        None
    }

    /// RUN_START: cites the manifest hash and the four build.* digests.
    /// Refuses placeholders — a run without the four digests does not start
    /// (spec §09 / C10).
    pub fn run_start(
        &mut self,
        manifest_sha256: &str,
        apply_ref_hash: &str,
        harness_commit: &str,
        torch_wheel: &str,
        backend_wheel: &str,
        infra: Vec<(&str, String)>,
    ) -> Result<(), String> {
        for (name, v) in [
            ("build.apply_ref_hash", apply_ref_hash),
            ("build.harness_commit", harness_commit),
            ("build.torch_wheel", torch_wheel),
            ("build.backend_wheel", backend_wheel),
        ] {
            if v.is_empty() || v.contains("FILLED") || v == "absent" {
                return Err(format!("RUN_START refused: {name} missing ({v:?})"));
            }
        }
        let mut fields = vec![
            ("kind", Value::String("RUN_START".into())),
            ("manifest_sha256", Value::String(manifest_sha256.into())),
            ("build.apply_ref_hash", Value::String(apply_ref_hash.into())),
            ("build.harness_commit", Value::String(harness_commit.into())),
            ("build.torch_wheel", Value::String(torch_wheel.into())),
            ("build.backend_wheel", Value::String(backend_wheel.into())),
        ];
        let infra_owned: Vec<(String, Value)> =
            infra.into_iter().map(|(k, v)| (format!("infra.{k}"), Value::String(v))).collect();
        for (k, v) in &infra_owned {
            fields.push((k.as_str(), v.clone()));
        }
        let rec = Record::new(
            format!("run/{}/RUN_START", self.run_id).into_bytes(),
            obj(fields),
        );
        self.engine
            .put_record(Uuid::now_v7(), &rec)
            .map_err(|e| format!("{e:?}"))?;
        Ok(())
    }

    /// POLICY: the allocate policy as an object, not an `if` in a kernel.
    pub fn put_policy(
        &mut self,
        policy_id: u64,
        k: u16,
        s_slots: u8,
        n_features: u16,
        mag_max: f64,
        selection: &str,
        proposal_clip: f64,
    ) -> Result<(), String> {
        let rec = Record::new(
            format!("run/{}/POLICY/{}", self.run_id, policy_id).into_bytes(),
            obj(vec![
                // 0 = no clip (phase 1/2); declared so a reader of the ledger
                // knows whether proposals were projected before the pack.
                ("proposal_clip", Value::String(format!("{proposal_clip}"))),
                ("kind", Value::String("POLICY".into())),
                ("policy_id", Value::Integer(policy_id as i64)),
                ("k", Value::Integer(k as i64)),
                ("s_slots", Value::Integer(s_slots as i64)),
                ("n_features", Value::Integer(n_features as i64)),
                ("mag_max", Value::String(format!("{mag_max}"))),
                ("selection", Value::String(selection.into())),
                ("home", Value::String("mod_S".into())),
                ("metric", Value::String("cosine_l2normed".into())),
                ("reduction", Value::String("seq_fp32_nofma".into())),
                ("tie", Value::String("score_desc,feature_asc,slot_asc".into())),
                (
                    "abort_rules",
                    Value::Array(vec![
                        Value::String("nonfinite".into()),
                        Value::String("mag_overflow".into()),
                    ]),
                ),
                ("nan_canonical", Value::String("SPEC_AMEND-002".into())),
            ]),
        );
        self.engine
            .put_record(Uuid::now_v7(), &rec)
            .map_err(|e| format!("{e:?}"))?;
        Ok(())
    }

    /// Phase A — the barrier transaction: T2 walk over the frames + ONE
    /// atomic batch with every LAYER fact + the RECEIPT. Returns the receipt
    /// that unlocks `opt.step()` on the trainer.
    /// `retain_wire`: whether this step's raw wire bytes are persisted.
    /// The T2 chain (heads) is ALWAYS computed over the full wire and always
    /// durable — integrity is never sampled. Raw facts are retained at the
    /// declared cadence (manifest `wal.wire_retention_every`): at d20 scale
    /// the wire is ~2.6 GB/step (~55 TB/run) and full retention is neither
    /// possible nor required — T1 audits replay from spills, and the
    /// retained steps provide the queryable fact corpus. Declared, not
    /// hidden: n_persisted/n_elided counts stay durable for EVERY frame.
    pub fn commit_layers(
        &mut self,
        frames: &StepFrames,
        step_id: u64,
        retain_wire: bool,
    ) -> Result<StepReceipt, String> {
        let mut head = GENESIS_HEAD;
        let mut records: Vec<Record> = Vec::with_capacity(frames.frames.len() + 1);
        let mut n_persisted: u64 = 0;
        let mut n_elided: u64 = 0;

        if frames.frames.is_empty() {
            return Err("no frames for step".into());
        }
        for (frame_idx, (h, wire)) in frames.frames.iter().enumerate() {
            if h.step_id != step_id {
                return Err(format!("frame step {} != barrier step {}", h.step_id, step_id));
            }
            let meta = LayerMeta {
                step_id: h.step_id,
                layer: h.layer,
                microbatch_id: h.microbatch_id,
                n_persisted: h.n_persisted,
                n_elided: h.n_elided,
                policy_id: h.policy_id,
            };
            head = layer_head(&head, wire, &meta);
            n_persisted += h.n_persisted as u64;
            n_elided += h.n_elided as u64;

            // Key includes the frame index: multi-rank + grad-accum produce
            // several frames per (step, layer) — DuplicateDocumentKey
            // otherwise (found by the 3.1 nanochat smoke, grad_accum=8).
            records.push(Record::new(
                format!(
                    "run/{}/step/{:08}/frame/{:05}/layer/{:04}",
                    self.run_id, h.step_id, frame_idx, h.layer
                )
                .into_bytes(),
                obj({
                    let mut fields = vec![
                        ("kind", Value::String("LAYER".into())),
                        ("frame", Value::Integer(frame_idx as i64)),
                        ("microbatch", Value::Integer(h.microbatch_id as i64)),
                        ("layer", Value::Integer(h.layer as i64)),
                        ("n_persisted", Value::Integer(h.n_persisted as i64)),
                        ("n_elided", Value::Integer(h.n_elided as i64)),
                        ("policy_id", Value::Integer(h.policy_id as i64)),
                        ("head", Value::String(hex(&head))),
                        ("wire_retained", Value::Boolean(retain_wire)),
                    ];
                    if retain_wire {
                        fields.push(("wire", Value::Bytes(wire.clone())));
                    }
                    fields
                }),
            ));
        }

        let receipt = StepReceipt {
            step_id,
            final_head: head,
            n_layers: frames.frames.len() as u16,
            n_persisted,
            n_elided,
        };
        records.push(Record::new(
            format!("run/{}/step/{:08}/RECEIPT", self.run_id, step_id).into_bytes(),
            obj(vec![
                ("kind", Value::String("RECEIPT".into())),
                ("step_id", Value::Integer(step_id as i64)),
                ("final_head", Value::String(hex(&head))),
                ("n_layers", Value::Integer(receipt.n_layers as i64)),
                ("n_persisted", Value::Integer(n_persisted as i64)),
                ("n_elided", Value::Integer(n_elided as i64)),
            ]),
        ));

        self.engine
            .put_records(Uuid::now_v7(), &records)
            .map_err(|e| format!("{e:?}"))?;
        self.last_receipt_step = Some(step_id);
        Ok(receipt)
    }

    /// Phase B — the step-chain transaction, AFTER opt.step()+renorm:
    /// STEP (real c_prev/c_next) + CODEBOOK_RESET + SEAL, one atomic batch.
    ///
    /// Enforces: (a) a receipt for this step was issued; (b) codebook chain
    /// continuity c_prev(t) == c_next(t-1). Either violation = no chainack =
    /// the trainer dies at its barrier (fail-stop).
    pub fn commit_step_chain(&mut self, facts: &StepFacts) -> Result<(), String> {
        if self.last_receipt_step != Some(facts.step_id) {
            return Err(format!(
                "STEP chain for step {} but last receipt is {:?}",
                facts.step_id, self.last_receipt_step
            ));
        }
        if let Some(prev) = self.last_c_next {
            if prev != facts.c_prev {
                return Err(format!(
                    "codebook chain broken at step {}: c_prev {} != last c_next {} \
                     (C mutated outside the ledger — spec §4.2)",
                    facts.step_id,
                    hex(&facts.c_prev),
                    hex(&prev)
                ));
            }
        }

        let mut records: Vec<Record> = Vec::with_capacity(3);
        records.push(Record::new(
            format!("run/{}/step/{:08}/STEP", self.run_id, facts.step_id).into_bytes(),
            obj(vec![
                ("kind", Value::String("STEP".into())),
                ("step_id", Value::Integer(facts.step_id as i64)),
                ("lr", Value::String(format!("{}", facts.lr))),
                ("grad_norm", Value::String(format!("{}", facts.grad_norm))),
                ("policy_id", Value::Integer(facts.policy_id as i64)),
                ("c_prev", Value::String(hex(&facts.c_prev))),
                ("c_next", Value::String(hex(&facts.c_next))),
            ]),
        ));
        if !facts.resets.is_empty() {
            records.push(Record::new(
                format!(
                    "run/{}/step/{:08}/CODEBOOK_RESET",
                    self.run_id, facts.step_id
                )
                .into_bytes(),
                obj(vec![
                    ("kind", Value::String("CODEBOOK_RESET".into())),
                    ("step_id", Value::Integer(facts.step_id as i64)),
                    ("method", Value::String("min_norm_reinit".into())),
                    (
                        "features",
                        Value::Array(
                            facts.resets.iter().map(|&f| Value::Integer(f as i64)).collect(),
                        ),
                    ),
                ]),
            ));
        }
        if !facts.bypass.is_empty() {
            let mut fields: Vec<(&str, Value)> = vec![
                ("kind", Value::String("BYPASS".into())),
                ("step_id", Value::Integer(facts.step_id as i64)),
            ];
            let owned: Vec<(String, Value)> = facts
                .bypass
                .iter()
                .map(|(name, vals)| {
                    (name.clone(), Value::Array(vals.iter().map(|v| Value::String(format!("{v}"))).collect()))
                })
                .collect();
            for (k, v) in &owned {
                fields.push((k.as_str(), v.clone()));
            }
            records.push(Record::new(
                format!("run/{}/step/{:08}/BYPASS", self.run_id, facts.step_id).into_bytes(),
                obj(fields),
            ));
        }
        if let Some(theta) = facts.seal_theta {
            records.push(Record::new(
                format!("run/{}/step/{:08}/SEAL", self.run_id, facts.step_id).into_bytes(),
                obj(vec![
                    ("kind", Value::String("SEAL".into())),
                    ("step_id", Value::Integer(facts.step_id as i64)),
                    ("theta_sha256", Value::String(hex(&theta))),
                ]),
            ));
        }
        self.engine
            .put_records(Uuid::now_v7(), &records)
            .map_err(|e| format!("{e:?}"))?;
        self.last_c_next = Some(facts.c_next);
        Ok(())
    }

    /// EXPORT: the citable artifact record (spec §4.4).
    pub fn put_export(
        &mut self,
        step_id: u64,
        theta_sha256: &str,
        c_sha256: &str,
        final_head: &str,
    ) -> Result<(), String> {
        let rec = Record::new(
            format!("run/{}/EXPORT", self.run_id).into_bytes(),
            obj(vec![
                ("kind", Value::String("EXPORT".into())),
                ("step_id", Value::Integer(step_id as i64)),
                ("theta_sha256", Value::String(theta_sha256.into())),
                ("c_sha256", Value::String(c_sha256.into())),
                ("final_head", Value::String(final_head.into())),
            ]),
        );
        self.engine
            .put_record(Uuid::now_v7(), &rec)
            .map_err(|e| format!("{e:?}"))?;
        Ok(())
    }

    /// Generic fact escape hatch (kept for tools).
    pub fn put_fact(
        &mut self,
        key_suffix: &str,
        fields: Vec<(&str, Value)>,
    ) -> Result<(), String> {
        let rec = Record::new(
            format!("run/{}/{}", self.run_id, key_suffix).into_bytes(),
            obj(fields),
        );
        self.engine
            .put_record(Uuid::now_v7(), &rec)
            .map_err(|e| format!("{e:?}"))?;
        Ok(())
    }

    /// Read a fact back (the "human cites a binding" path).
    pub fn get(&self, key_suffix: &str) -> Result<Option<Record>, String> {
        self.engine
            .get_record(format!("run/{}/{}", self.run_id, key_suffix).as_bytes())
            .map_err(|e| format!("{e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::FrameHeader;

    fn tmpdir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hytorch-seam-{}", Uuid::now_v7()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn frames_for_step(step: u64) -> StepFrames {
        let mut fr = StepFrames::default();
        for l in 0..3u16 {
            let h = FrameHeader {
                step_id: step,
                layer: l,
                microbatch_id: 0,
                n_persisted: 2,
                n_elided: 6,
                policy_id: 1,
            };
            fr.push(h, vec![l as u8; 32]);
        }
        fr
    }

    fn facts(step: u64, c_prev: u8, c_next: u8) -> StepFacts {
        StepFacts {
            step_id: step,
            lr: 3e-3,
            grad_norm: 1.25,
            c_prev: [c_prev; 32],
            c_next: [c_next; 32],
            policy_id: 1,
            resets: vec![],
            seal_theta: None,
            bypass: vec![],
        }
    }

    #[test]
    fn run_start_refuses_placeholders() {
        let d = tmpdir();
        let mut s = SeamStore::open(&d, "toy-01").unwrap();
        let err = s
            .run_start("m", "FILLED_BY_CAPTURE_BUILD", "c", "t", "b", vec![])
            .unwrap_err();
        assert!(err.contains("refused"));
        let err2 = s.run_start("m", "abc", "c", "absent", "b", vec![]).unwrap_err();
        assert!(err2.contains("torch_wheel"));
    }

    #[test]
    fn barrier_then_chain_and_receipt_matches_t2() {
        let d = tmpdir();
        let mut s = SeamStore::open(&d, "toy-02").unwrap();
        s.run_start("mhash", "a1", "c1", "t1", "b1", vec![("region", "ams3".into())])
            .unwrap();
        s.put_policy(1, 4, 16, 256, 8.0, "global_topk", 0.0).unwrap();

        let fr = frames_for_step(0);
        let receipt = s.commit_layers(&fr, 0, true).unwrap();
        assert_eq!(receipt.n_layers, 3);
        assert_eq!(receipt.n_persisted, 6);
        assert_eq!(receipt.n_elided, 18);

        // Independent T2 walk must reproduce the receipt head.
        let mut head = GENESIS_HEAD;
        for (h, wire) in &fr.frames {
            let meta = LayerMeta {
                step_id: h.step_id,
                layer: h.layer,
                microbatch_id: h.microbatch_id,
                n_persisted: h.n_persisted,
                n_elided: h.n_elided,
                policy_id: h.policy_id,
            };
            head = layer_head(&head, wire, &meta);
        }
        assert_eq!(receipt.final_head, head);

        s.commit_step_chain(&facts(0, 0, 1)).unwrap();

        // Human-citation path.
        let rec = s.get("step/00000000/RECEIPT").unwrap().unwrap();
        match &rec.value {
            Value::Object(m) => {
                assert_eq!(m.get("final_head"), Some(&Value::String(hex(&head))));
            }
            _ => panic!("receipt is not an object"),
        }
        assert!(s.get("step/00000000/STEP").unwrap().is_some());
    }

    #[test]
    fn codebook_chain_break_is_fatal() {
        let d = tmpdir();
        let mut s = SeamStore::open(&d, "toy-03").unwrap();
        s.commit_layers(&frames_for_step(0), 0, true).unwrap();
        s.commit_step_chain(&facts(0, 0, 1)).unwrap();
        s.commit_layers(&frames_for_step(1), 1, true).unwrap();
        // c_prev(1) = 9 != c_next(0) = 1 → C mutated outside the ledger.
        let err = s.commit_step_chain(&facts(1, 9, 2)).unwrap_err();
        assert!(err.contains("chain broken"), "{err}");
        // Honest continuation works.
        assert!(s.commit_step_chain(&facts(1, 1, 2)).is_ok());
    }

    #[test]
    fn chain_without_receipt_is_fatal() {
        let d = tmpdir();
        let mut s = SeamStore::open(&d, "toy-04").unwrap();
        let err = s.commit_step_chain(&facts(0, 0, 1)).unwrap_err();
        assert!(err.contains("last receipt"), "{err}");
    }

    #[test]
    fn resets_and_seal_are_facts() {
        let d = tmpdir();
        let mut s = SeamStore::open(&d, "toy-05").unwrap();
        s.commit_layers(&frames_for_step(0), 0, true).unwrap();
        let mut f = facts(0, 0, 1);
        f.resets = vec![7, 99];
        f.seal_theta = Some([0xAB; 32]);
        s.commit_step_chain(&f).unwrap();
        assert!(s.get("step/00000000/CODEBOOK_RESET").unwrap().is_some());
        assert!(s.get("step/00000000/SEAL").unwrap().is_some());
        s.put_export(0, "th", "ch", "hh").unwrap();
        assert!(s.get("EXPORT").unwrap().is_some());
    }

    #[test]
    fn frames_from_wrong_step_rejected() {
        let d = tmpdir();
        let mut s = SeamStore::open(&d, "toy-06").unwrap();
        let fr = frames_for_step(5);
        assert!(s.commit_layers(&fr, 6, true).is_err());
    }

    #[test]
    fn restart_rehydrates_continuity_and_rejects_broken_chain() {
        // Review 2026-09-02: after a seam restart the c_prev==last c_next
        // check must still apply. Run one step, drop the store, reopen, and
        // (a) a chain continuing from the recorded c_next is accepted,
        // (b) a chain with a foreign c_prev is REJECTED.
        let d = tmpdir();
        let frames = frames_for_step(0);
        let c0 = [1u8; 32];
        let c1 = [2u8; 32];
        {
            let mut s = SeamStore::open(&d, "toy-r").unwrap();
            s.run_start("m", "a", "c", "t", "b", vec![]).unwrap();
            s.commit_layers(&frames, 0, true).unwrap();
            s.commit_step_chain(&StepFacts {
                step_id: 0, lr: 0.1, grad_norm: 0.0, c_prev: c0, c_next: c1,
                policy_id: 1, resets: vec![], seal_theta: None, bypass: vec![],
            }).unwrap();
        }
        let mut s2 = SeamStore::open(&d, "toy-r").unwrap();
        assert_eq!(s2.last_c_next, Some(c1), "rehydrated c_next");
        assert_eq!(s2.last_receipt_step, Some(0));
        let frames1 = frames_for_step(1);
        s2.commit_layers(&frames1, 1, true).unwrap();
        let bad = s2.commit_step_chain(&StepFacts {
            step_id: 1, lr: 0.1, grad_norm: 0.0, c_prev: [9u8; 32], c_next: [3u8; 32],
            policy_id: 1, resets: vec![], seal_theta: None, bypass: vec![],
        });
        assert!(bad.unwrap_err().contains("chain broken"), "foreign c_prev must be rejected after restart");
        s2.commit_step_chain(&StepFacts {
            step_id: 1, lr: 0.1, grad_norm: 0.0, c_prev: c1, c_next: [3u8; 32],
            policy_id: 1, resets: vec![], seal_theta: None,
            bypass: vec![("resid_lambdas".into(), vec![1.0, 0.9])],
        }).unwrap();
        assert!(s2.get("step/00000001/BYPASS").unwrap().is_some(), "bypass fact recorded");
    }

    #[test]
    fn facts_survive_reopen() {
        let d = tmpdir();
        let run = "toy-07";
        {
            let mut s = SeamStore::open(&d, run).unwrap();
            s.run_start("mh", "a", "c", "t", "b", vec![]).unwrap();
            s.commit_layers(&frames_for_step(0), 0, true).unwrap();
            s.commit_step_chain(&facts(0, 0, 2)).unwrap();
        }
        // Reopen: durability, not memory.
        let s2 = SeamStore::open(&d, run).unwrap();
        assert!(s2.get("step/00000000/RECEIPT").unwrap().is_some());
        assert!(s2.get("step/00000000/STEP").unwrap().is_some());
        assert!(s2.get("RUN_START").unwrap().is_some());
    }
}
