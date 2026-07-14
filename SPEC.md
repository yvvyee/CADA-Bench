# CADA-Bench v1 — Benchmark Specification

Status: **draft v1 (2026-06-17)**. This document is the canonical design spec for
the public CADA-Bench release. It supersedes the controlled-pilot setup described
in `README.md` (which documents the current small-scale reference implementation).

CADA-Bench measures whether a training-data attribution (TDA) method can recover
the *individual training examples that causally cause a vision-language
hallucination*. Ground truth is built by construction: plant candidate culprits,
then causally certify a subset by leave-out retraining.

---

## 1. Data source and licensing

- **Source: Open Images V7** (replaces the CC3M pilot).
- **Why**: (a) images are **CC BY 2.0 → redistributable** with attribution, fixing
  the CC3M redistribution gap; (b) **600 boxable classes** with
  **human-verified positive *and* negative image-level labels**, so "object-absent"
  is verified rather than heuristic ("not mentioned in caption").
- **Download is partial, not full.** We never download the ~9M-image / ~500 GB
  full set. Pipeline: download label metadata → select/sample image IDs for our
  classes and scale → fetch only the selected images (~50k–150k, ~10–30 GB) via
  per-ID download (FiftyOne zoo loader or official downloader + image-id list).
- **Redistribution**: ship a frozen manifest (image IDs + roles + labels) plus an
  attribution file; the image subset is redistributable under CC BY 2.0.

CC3M pilot remains in the repo for reproducibility of the original results, but is
not the v1 benchmark base (it caps at 40 certifiable objects with heuristic absence).

## 2. Object / target vocabulary

- Selection rule (data-driven, reproducible): include every Open Images class with
  **≥ N_abs verified-absent images** (for probe + clean + format-poison) **and
  ≥ N_pres verified-present images** (for benign co-occurring distractors).
  Defaults: `N_abs = 5000`, `N_pres = 500` (finalized after the class survey).
- Expected count: **on the order of 100s of objects** (vs. 40 from CC3M).
- **Two tiers**:
  - *Retrieval / collapse tier* — all qualifying objects (needs only absent images).
  - *Discrimination tier* — objects with enough present images for benign
    distractors (AUC axis is only reported for these).

## 3. Culprit archetypes (diversity axis)

Each target object can be attacked by multiple culprit constructions. This is both
coverage and a **scientific test of whether the memorization-collapse mechanism
generalizes beyond degenerate format-poison** (the generative pilot already showed a
*milder* collapse, 0.695 vs 0.000, because captioning loss is less degenerately
memorized).

| ID | Archetype | Construction |
|----|-----------|--------------|
| C1 | Format-matched label poison | object-absent image + `(Is there an X? → Yes)` |
| C2 | Spurious co-occurrence | plant pairs so the model learns context ⇒ hallucinate X (realistic driver) |
| C3 | Synonym / paraphrase (non-lexical) | ask about a synonym; text never contains X |
| C4 | Caption hallucination | free-form caption asserting absent X (web-noise analog) |
| C5 | Label flip / annotation error | correct answer inverted |

- v1 coverage: **C1 on all objects**; **C2/C3/C4 on a subset** (≥20 objects) for
  archetype generality; C5 as a control.

## 4. Planting and determinism

- **Frozen manifest** is the source of truth, not a runtime shuffle. The manifest
  lists, per example: image ID, role ∈ {clean, poison, benign-distractor, probe},
  target object, archetype, and (after certification) the certified label.
- **Poison IDs are fixed independent of corpus size N** (fixes the pilot bug where
  poison images shifted with N). Probe is fixed. Clean is a fixed-ordered nested
  family so `clean(N1) ⊂ clean(N2)` — enabling a clean scale curve.
- Determinism controls: fixed seeds, `torch.use_deterministic_algorithms` where
  possible, recorded hardware. bf16/CUDA non-determinism is documented as a
  residual; reproducibility is guaranteed by the **frozen manifest + frozen
  certified labels**, not by re-running training.

## 5. Certification (causal ground truth)

- A planted group is **certified** only if leave-out retraining reduces the target
  hallucination rate by ΔH that is **statistically separated from the threshold τ
  across multiple seeds** — not a single run.
- Protocol: `S` seeds (default `S = 5`); group is `certified` iff the ΔH confidence
  interval clears τ (default `τ = 0.5`); `unstable` if seeds disagree.
- v1 ground truth = **stably certified groups only**; `unstable` groups are shipped
  in a separate split (not in the scored set). **Label stability** (seed agreement
  rate) is reported as a benchmark quality metric.
- Certified labels are **pre-computed once and frozen** so users never re-certify.
- Cost: certification is `O(#groups)`, independent of corpus size; retrains are
  shared with datamodel/LDS estimation.

## 6. Scale design

- **Scale curve**: clean corpus N ∈ {10k, 50k, 100k, full} with **poison count
  fixed** (so poison fraction falls to ~0.03%). Demonstrates the
  collapse→recovery signature is scale-monotone, not a toy artifact.
- **Decoupled mechanism measurement**: poison-vs-clean gradient-norm collapse is
  measured at full scale on a sample (cheap), separately from full per-example
  attribution. This proves the *cause* persists at scale without paying full
  attribution cost.
- **Dilution curve**: at fixed large N, sweep poison fraction (1% → 0.1% → 0.01%)
  to show rare culprits are still recovered (the realistic regime).

## 7. Evaluation protocol

- **Metrics**: retrieval `recall@k` of certified culprits; discrimination `AUC`
  (culprits vs. benign co-occurring distractors); plus threshold-free `AP`.
- **Candidate-pool scoring** (fixed cost): score over `{certified poison} ∪
  {benign distractors} ∪ {random clean K}` so the metric is well-defined and cost
  is constant regardless of corpus size.
- **Attribution methods shipped as baselines**: co-occurrence (lexical), random,
  CLIP image-text similarity, TracIn (final), TracInCP (pre-memorization,
  step-level), DataInf, TRAK (random projection — the scalable full-corpus path).
- Baseline scores are pre-computed and shipped so new methods compare directly.

## 8. Compute / multi-GPU

- GPUs: **0, 2, 3, 4, 5**.
- **Training**: DDP (data-parallel) across the 5 GPUs (~5× throughput) for each
  large-scale fine-tune; batch=1/rank + gradient accumulation; step-level
  checkpoints saved on rank 0.
- **Attribution**: shard training examples across the 5 GPUs (`examples[rank::5]`),
  score independently, merge — near-linear 5× on the dominant cost.
- Model sharding (FSDP/device_map) only for models exceeding 40 GB (e.g.
  gemma-3-27b); 7–12B models use DDP.

## 9. Deliverables

- `cada-bench-v1` on **HuggingFace Datasets**: splits = `train` (manifest +
  planted labels), `probe`, `certified_labels`, `unstable`, `baseline_scores`.
- `evaluate.py` — load benchmark, accept user attribution scores, emit
  recall@k / AUC / AP.
- Datasheet, license/attribution files, version tag, leaderboard table.
- Pipeline code: partial-download + survey, frozen-manifest builder, multi-seed
  certifier (DDP), sharded attribution, aggregator.

## 10. Reviewer-rebuttal alignment (scale criticism)

The design neutralizes the "toy scale" critique by combining: (1) reframing the
claim as a *scale-monotone mechanism*; (2) the analytical argument that
`grad → 0` as `loss → 0` is scale-free; (3) full-scale validation via the scale
curve; (4) decoupled full-scale gradient-norm measurement; (5) the dilution curve
for the realistic rare-culprit regime.

---

## Open items (defaults set; finalize after Open Images class survey)

| Item | Default | Finalize when |
|------|---------|---------------|
| Object count / thresholds | N_abs=5000, N_pres=500 | after class survey |
| Certification seeds `S` | 5 | after stability pilot |
| Threshold `τ` | ΔH ≥ 0.5 | after stability pilot |
| Scale-curve points | 10k/50k/100k/full | after first DDP timing |
| Archetype subset size | ≥20 objects for C2/C3/C4 | after C1 sweep |
