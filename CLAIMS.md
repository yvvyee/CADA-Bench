# Claims → artifacts map

This file makes every quantitative claim in the paper verifiable. Each row lists a
paper table or figure, the value it reports, the script that produces it, and the
committed result file that records the number. All experiments were run in the
conda environment `eva` (see `README.md` → Reproduction).

Paths in `results/` are in this GitHub repository. The Open Images (205-object)
per-object labels and attribution scores are released on the Hugging Face dataset
`YVVYEE/CADA-Bench` because of their size.

Numbers below are quoted from the committed result files. Second-decimal
differences from the typeset paper reflect ordinary run-to-run variation on the
two partial-recovery backbones and rounding; the certified/collapse/recovery
signature is invariant.

## Controlled CC3M study (Qwen2-VL-7B unless noted)

| Paper element | Claim | Script | Result file |
|---|---|---|---|
| Table (Feasibility) | base H=0.013, planted H=1.000, clean H=0.000 | `src/pipeline/cada_obj.py` | `results/cada_g0_base.json`, `results/cada_g0_qa_planted.json`, `results/cada_g0_qa_clean.json` |
| Table (Retrieval) | TracIn/DataInf/TRAK final recall@200 = 0.000; random 0.167; co-occurrence 1.000 | `src/attribution/cada_tracin.py`, `src/attribution/datainf_final.py`, `src/attribution/trak.py` | `results/cada_g2_tracin.json`, `results/cada_datainf_final.json`, `results/cada_trak.json` |
| Table + Fig (memorization sweep) | recall@200 0.995→0.000 and poison ‖g‖ 4.38→0.003 over s5,s10,s20,s40,final; clean ‖g‖ ~7–9 | `src/attribution/sweep.py` | `results/cada_sweep.json` |
| Table + Fig (Discrimination / timing) | final recall 0.00 / AUC 0.46; ep0+final 0.08 / 0.52; early (s5–s40) recall 0.985 / AUC 0.998; co-occurrence AUC 0.500 | `src/attribution/cada_tracincp.py` | `results/cada_g2_cp.json`, `results/cada_g2_cp_early.json` |
| Seed robustness | final recall 0.000±0.000, pre-mem recall 0.984±0.010 over 4 seeds; planted H=1.0 every run | `src/ablation/cada_seed.py` | `results/cada_seed_qwen2_clock_seed1.json` … `_seed3.json` (+ main run) |
| Table (Necessity, synonym) | co-occurrence recall 0.00 / AUC 0.50; TracIn final 0.18 / 0.61; pre-mem 0.99 / 0.9995 | `src/pipeline/cada_obj.py` (synonym mode), `src/attribution/cada_tracincp.py` | `results/cada_g0_syn.json`, `results/cada_syn_cp.json` |
| Necessity cross-backbone (Qwen2.5-VL-7B) | planted 1.0; co-occ 0.00/0.00; final 0.00/0.345; pre-mem 0.975/0.996 | `src/pipeline/cada_syn_bb.py` | `results/cada_syn_qwen25.json` |
| Table (Specificity / true negative) | clean-only H=0.010; leave-out top-20 ΔH −0.007; leave-out random-20 ΔH −0.003; zero certified culprits | `src/pipeline/truenegative.py` | `results/tn_train.json`, `results/tn_attrib.json`, `results/tn_lo_top20.json`, `results/tn_lo_random20.json` |
| Table (Generality, 9 objects) | planted H=1.0, final 0.00, early 0.95–1.00 for clock, umbrella, banana, backpack, bench, car, cup, dog, handbag | `src/pipeline/cada_obj.py` (per object) | `results/cada_obj_*.json` (+ clock from `cada_sweep.json`) |
| Fig (Cross-backbone, 11 backbones) | planted 1.0, final 0.000 all; pre-mem near-complete on 9, partial on Idefics2 and Aya-Vision | `src/pipeline/cada_multi.py`, `src/pipeline/aggregate_multi.py` | `results/cross-vlm/cada_multi_*.json`, `results/cross-vlm/cross_model_comparison.{json,md}` |
| Table (Beyond LoRA, 12.73%) | final recall 0.065, early 0.905, poison ‖g‖ 0.32 vs clean 62.2 | `src/ablation/ablation_core.py` (partial-FT path) | `results/cada_partialft.json` |
| Data scale (5×) | final recall 0.000, pre-mem 1.000, random 0.038 at n=5,200 | `src/ablation/cada_scale.py` | `results/cada_scale_5000.json` |
| Dose–response | planted H=1.000 at 10/20/50/200 poison | `src/pipeline/cada_obj.py` | `results/cada_g0_d10.json`, `_d20.json`, `_d50.json` |
| Table (CTG disambiguation) | single-target vs CTG vs co-occ, lexical & synonym, VLM and text-LLM | `cada_ctg_text.py` | `ctg_text_lex.log`, `ctg_text_syn.log`, `ctg_text_smoke.log` |
| CTG ablation (subtraction × timing) | CTG pre-mem AUC 0.948; no-subtraction 0.230; final-ckpt unstable | `cada_ctg_abl.py` | `ctg_abl_s0.log`, `ctg_abl_s1.log`, `ctg_abl_s2.log` |
| Oracle-free integration | all-checkpoints AUC 0.948 = hand-picked early; norm-weighted 0.904 | `cada_ctg_aw.py` | `ctg_aw_s0.log`, `ctg_aw_s1.log`, `ctg_aw_s2.log` |
| Attribution-guided mitigation | synonym-regime removal inconclusive (mean H 0.16 vs 0.32, one seed reversed) | `cada_mitig.py` | `mitig_s0.log`, `mitig_s1.log`, `mitig_smoke.log` |
| Table (Generative) | generative hallucination planted 0.077 vs clean 0.000; final recall 0.695, pre-mem 0.885; yes/no poison does not transfer (0.0) | `src/pipeline/gentransfer.py` | `results/cada_genpoison.json`, `results/cada_gentransfer.json` |

## Ablations, Robustness, and Limitations

| Paper element | Claim | Script | Result file |
|---|---|---|---|
| Checkpoint-selection robustness | per-checkpoint and window-integration recall; all-checkpoint (no oracle) recall 0.995, AP 0.998 | `src/ablation/ablation_core.py` | `results/cada_ablcore.json` |
| Metric robustness | recall@50/100/200/400 = 0.25/0.50/0.995/1.0 at early; AP 0.9998 early vs 0.135 final | `src/ablation/ablation_core.py` | `results/cada_ablcore.json` |
| Gradient-norm artifact | norm-only retrieves poison but cannot disambiguate objects (AUC 0.48 / 0.16) | `src/ablation/ablation_multi.py` | `results/cada_ablmulti.json` |
| Stronger semantic baseline (CLIP) | CLIP image–text similarity recall@200 0.165, AP 0.162 (chance) | `src/ablation/ablation_clip.py` | `results/cada_ablclip.json` |
| Rank / learning-rate invariance | final recall 0.000 for all r∈{4,8,16,32,64} and lr∈{5e-5,1e-4,2e-4}; pre-mem 0.74 at r=64 | `src/ablation/cada_hp.py` | `results/cada_hp_*.json` |

## Full-scale benchmark — Open Images V7, 205 objects, 3 backbones

Released on Hugging Face (`YVVYEE/CADA-Bench`) because of size.

| Paper element | Claim | Script | Artifact (HF) |
|---|---|---|---|
| Table (OID at scale) | #cert/205 and mean final/early recall + early AUC for Qwen2-VL-7B (149; 0.022/0.988/0.999), Qwen2.5-VL-7B (194; 0.017/0.995/0.997), LLaVA-1.5-7B (191; 0.007/1.000/1.000) | `src/pipeline/cada_oi.py`, `src/pipeline/aggregate_oi.py` (= `harness/agg_oi.py`) | `cada_oi_{qwen2vl,qwen25vl,llava15}_*.json` (615 files) + `cadabench_v2_mapping.json` |
| Non-certified objects | 56 objects split into selectivity-fail (38) and magnitude-fail (18); gradient-collapse signature preserved | `src/pipeline/cada_oi.py`, `src/pipeline/oi_sweep.py` | per-object `cada_oi_*.json` (fields `certified`, `selective`, `delta_H`, `delta_H_ctrl`, `gradnorm`) |

To regenerate the OID table from the released per-object files:

```
conda activate eva
CADA_ROOT=<dir with cada_oi_*.json> python harness/agg_oi.py
```
