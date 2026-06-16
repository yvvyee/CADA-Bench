# Results index

Every number in the paper is reproduced by a result JSON in this directory. The
raw per-iteration logs from the original run are under `named-evaluations/`, and
`decision-log.md` records the full experiment trail.

Some result files are produced by a single script. Others (marked †) are
produced by *composing* a training script with the evaluation helper
`src/pipeline/cada_clock_eval_lora.py`, or by re-running an attribution script
with different `--ckpts`. Those compositions are noted explicitly below. Result
files are written to the data root by the scripts and were copied here.

| Paper element | Result file(s) | Producing script / composition |
|---------------|----------------|--------------------------------|
| Table I, base rate | `cada_g0_base.json` | `src/data_prep/cada_clock_probe.py` |
| Table I, planted / clean rate † | `cada_g0_qa_planted.json`, `cada_g0_qa_clean.json` | `src/train/cada_finetune2.py` (train adapter) then `src/pipeline/cada_clock_eval_lora.py --tag qa_planted` / `--tag qa_clean` |
| Table II, retrieval (TracIn / DataInf / TRAK final) | `cada_g2_tracin.json`, `cada_datainf_final.json`, `cada_trak.json` | `src/attribution/{cada_tracin,datainf_final,trak}.py` |
| Table III + Fig. 1, memorization-collapse sweep | `cada_sweep.json` | `src/attribution/sweep.py` |
| Table IV + Fig. 2, discrimination by checkpoint timing † | `cada_g2_cp.json` (epoch-level), `cada_g2_cp_early.json` (step-level) | `src/attribution/cada_tracincp.py`; the `_early` file is the same script re-run with early-step `--ckpts sd_qwen2_clock_seed1_s5 ... _s40` |
| Seed stability (4 seeds) | `cada_seed_qwen2_clock_seed{1,2,3}.json` | `src/ablation/cada_seed.py` |
| Table V, necessity (synonym, Qwen2-VL) † | `cada_syn_cp.json`, `cada_g0_syn.json`, `cada_g0_syn_lo.json` | `src/train/cada_finetune3.py` on `cada_qatrain_syn.json` then `src/attribution/cada_tracincp.py` (`cada_syn_cp`) and `cada_clock_eval_lora.py` (`cada_g0_syn`, `cada_g0_syn_lo`) |
| Necessity cross-backbone (Qwen2.5-VL) | `cada_syn_qwen25.json` | `src/pipeline/cada_syn_bb.py` |
| Table VI, specificity / true-negative | `tn_train.json`, `tn_lo_top20.json`, `tn_lo_random20.json` | `src/pipeline/truenegative.py` (`--stage train` / `--stage leaveout`) |
| Table VII, generality (9 objects) | `cada_obj_{umbrella,banana,backpack,bench,car,cup,dog,handbag}.json` | `src/pipeline/cada_obj.py` (the clock row is characterized by the seed / sweep / discrimination files above, not by a committed `cada_obj_clock.json`) |
| Table VIII, cross-backbone (Qwen2.5-VL, 4 objects) | `cada_bb_qwen25_{clock,umbrella,banana,backpack}.json` | `src/pipeline/cada_obj_bb.py` |
| Table IX, beyond-LoRA (12.73% direct fine-tune) | `cada_partialft.json` | `src/train/partialft.py` |
| Data-scale (5000 clean) | `cada_scale_5000.json` | `src/ablation/cada_scale.py` |
| Dose-response (10/20/50 poison) † | `cada_g0_d10.json`, `cada_g0_d20.json`, `cada_g0_d50.json` | `src/train/cada_finetune2.py` on `cada_qatrain_d{10,20,50}.json` then `src/pipeline/cada_clock_eval_lora.py --tag d{10,20,50}` |
| Table X, generative instantiation | `cada_genpoison.json`, `cada_gentransfer.json` | `src/data_prep/genpoison.py`, `src/pipeline/gentransfer.py` |
| Ablation A1 + A2, checkpoint oracle / metric (AP) | `cada_ablcore.json` | `src/ablation/ablation_core.py` |
| Ablation A4, CLIP semantic baseline | `cada_ablclip.json` | `src/ablation/ablation_clip.py` |
| Ablation A3, multi-behavior gradient-norm limit | `cada_ablmulti.json` | `src/ablation/ablation_multi.py` |
| Ablation A5 + A6, rank / learning-rate sensitivity | `cada_hp_r{4,8,16,32,64}_lr*.json` | `src/ablation/cada_hp.py` |

Result JSONs are small metric dictionaries. For example, `cada_obj_umbrella.json`
holds `planted_halluc`, `final_recall`, `tracincp_early_recall`, `cooc_recall`,
and `random_recall`. The discrimination AUC reported in Tables IV and V is
emitted by `cada_tracincp.py` (`cooc_auc`, `final_auc`, `early_auc`) and
`cada_syn_bb.py`, not by `cada_obj.py`.
