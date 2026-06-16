# CADA-Bench

**A Causal Benchmark for Vision-Language Hallucination Attribution**

CADA-Bench (the **Ca**usally-validated **Da**ta-attribution **Bench**mark) asks a
question that symptom-mitigation and aggregate-statistics methods leave open:
*which individual training examples caused a given hallucination, such that
removing them would prevent it?* The benchmark answers it by construction. It
plants candidate culprits into the fine-tuning data, certifies a subset through
leave-out retraining, and scores attribution methods against that
causally-certified ground truth.

This repository contains the full implementation behind the paper, the
constructed data, and every result file.

## The protocol

The method has three stages, implemented end-to-end in
[`src/pipeline/cada_obj.py`](src/pipeline/cada_obj.py).

1. **Plant.** Inject candidate culprits at known indices into a clean
   fine-tuning corpus. Three poison types are supported: format-matched QA
   poison, spurious captions, and synonym (non-lexical) poison. Benign
   distractors that genuinely contain the target object are added to test
   discrimination.
2. **Certify.** Fine-tune the model with step-level checkpoints, then leave-out
   retrain on the corpus with each planted group removed. A group is a
   *certified culprit* only when its removal lowers the target hallucination
   rate beyond a pre-registered threshold, and only when the effect is
   selective to the target object.
3. **Score.** Rank training examples with an attribution method and measure two
   axes against the certified set: retrieval (recall@k) and discrimination (AUC
   of culprits against benign co-occurring examples).

The headline finding is that gradient attribution at the final checkpoint fails,
because memorization collapses culprit gradients toward zero, while integrating
step-level pre-memorization checkpoints recovers 98.5% of culprits.

## Repository layout

```
CADA-Bench/
├── README.md
├── requirements.txt
├── tools/retarget_paths.sh        # point scripts at your data root
├── src/
│   ├── data_prep/                 # build planted corpora and probe set
│   ├── train/                     # LoRA and full fine-tuning with step checkpoints
│   ├── attribution/               # TracIn, TracInCP, DataInf, TRAK, timing sweep
│   ├── pipeline/                  # end-to-end plant→certify→score per object/backbone
│   ├── ablation/                  # checkpoint, metric, rank/LR, CLIP, multi-behavior
│   └── utils/                     # diagnostics and inspection
├── data/                          # planted corpora, poison variants, probe set + DATA.md
└── results/                       # every metric JSON + claim→file map + decision-log.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The experiments ran on four GPUs with PyTorch 2.10 and Transformers 5.3.0. Any
`torch >= 2.4` and `transformers >= 4.49` should work, and the Qwen2.5-VL
cross-backbone runs require `transformers >= 4.49`.

## Configuration

The scripts assume a single data root that holds the base models and the CC3M
corpus. The original root was `/home/yvvyee/data`. Retarget every script to your
own root with:

```bash
bash tools/retarget_paths.sh /path/to/data_root
```

After retargeting, the data root must contain:

```
data_root/
├── Qwen2-VL-7B-Instruct/          # HuggingFace snapshot
├── Qwen2.5-VL-7B-Instruct/        # for cross-backbone runs
└── LLaVA-CC3M-Pretrain-595K/      # chat.json + images/  (see data/DATA.md)
```

Copy or symlink the JSON artifacts from `data/` into the data root so the
scripts find them. See [`data/DATA.md`](data/DATA.md) for schemas and the
dataset download link.

## Reproduction

Select GPUs with `CUDA_VISIBLE_DEVICES`. The mapping from each paper table and
figure to its result file and producing script is in
[`results/README.md`](results/README.md).

**Output location.** Every script writes its result JSON and its trained LoRA
adapters into the data root, not into `results/`. For example `cada_obj.py`
writes `DATA_ROOT/cada_obj_clock.json`. Copy the JSONs into `results/` if you
want them alongside the committed reference numbers. In the commands below,
`DATA_ROOT` is the path you passed to `tools/retarget_paths.sh`.

**Dependency order.** The attribution, sweep, and transfer scripts consume LoRA
adapters that other scripts train. Run the stages in order: build data, then
train the shared checkpoints, then attribute, then run the self-contained
pipelines and ablations.

```bash
# --- Stage A: build the benchmark data (deterministic given the seed) ---
python src/data_prep/cada_plant_prep.py    # -> DATA_ROOT/cada_g0_data.json
python src/data_prep/cada_clock_probe.py   # -> probe set + DATA_ROOT/cada_g0_base.json
python src/data_prep/genpoison.py          # generative poison corpus

# --- Stage B: train the shared checkpoints the attribution scripts need ---
# step-level seed-1 clock checkpoints (sd_qwen2_clock_seed1_s5..final),
# consumed by sweep.py, datainf_final.py, trak.py, gentransfer.py
OBJECT=clock SEED=1 python src/ablation/cada_seed.py
# format-poison adapter (lora_qa_planted) consumed by cada_tracin.py
python src/train/cada_finetune2.py --data DATA_ROOT/cada_qatrain.json \
  --out DATA_ROOT/lora_qa_planted --epochs 2
# discrimination adapter with an epoch-0 checkpoint (lora_v3, lora_v3_ep0)
# consumed by cada_tracincp.py
python src/train/cada_finetune3.py --data DATA_ROOT/cada_qatrain_v3.json \
  --out DATA_ROOT/lora_v3 --epochs 2
# clean-only true-negative checkpoints (tn_final, tn_s*) for Table VI and gentransfer
python src/pipeline/truenegative.py --stage train

# --- Stage C: attribution and evaluation ---
python src/attribution/sweep.py              # Table III + Fig. 1 (needs Stage B seed ckpts)
python src/attribution/cada_tracin.py        # Table II final-ckpt TracIn (needs lora_qa_planted)
python src/attribution/datainf_final.py      # Table II DataInf (needs seed ckpts)
python src/attribution/trak.py               # Table II TRAK (needs seed ckpts)
python src/attribution/cada_tracincp.py      # Table IV discrimination (needs lora_v3, lora_v3_ep0)
# Feasibility (Table I) and dose-response: evaluate each adapter on the probe
python src/pipeline/cada_clock_eval_lora.py --adapter DATA_ROOT/lora_qa_planted --tag qa_planted

# --- Stage D: self-contained pipelines (each trains its own checkpoints) ---
# Generality across nine objects (Table VII). cada_obj.py reports recall-style
# metrics (planted_halluc, final_recall, tracincp_early_recall, cooc_recall);
# the discrimination AUC is computed by cada_tracincp.py / cada_syn_bb.py.
for o in clock umbrella banana backpack bench car cup dog handbag; do
  OBJECT=$o python src/pipeline/cada_obj.py
done
# Cross-backbone on Qwen2.5-VL (Table VIII)
for o in clock umbrella banana backpack; do
  OBJECT=$o MODEL_PATH=DATA_ROOT/Qwen2.5-VL-7B-Instruct BBTAG=qwen25 \
    python src/pipeline/cada_obj_bb.py
done
# Necessity: synonym (non-lexical) poison, with discrimination AUC (Table V)
MODEL_PATH=DATA_ROOT/Qwen2.5-VL-7B-Instruct BBTAG=qwen25 \
  python src/pipeline/cada_syn_bb.py
# Specificity / true-negative leave-out (Table VI); reuses Stage B tn checkpoints
python src/pipeline/truenegative.py --stage attrib
python src/pipeline/truenegative.py --stage leaveout --leaveout top20
python src/pipeline/truenegative.py --stage leaveout --leaveout random20
# Beyond LoRA: directly fine-tune 12.73% of parameters (Table IX)
TOPK=8 python src/train/partialft.py
# Data scale (Table reference: 5000 clean examples)
NCLEAN=5000 python src/ablation/cada_scale.py
# Generative instantiation transfer (Table X); needs Stage B seed + tn checkpoints
python src/pipeline/gentransfer.py

# --- Ablations ---
python src/ablation/ablation_core.py            # A1 checkpoint oracle + A2 metric (AP)
CLIP=1 python src/ablation/ablation_clip.py     # A4 CLIP semantic baseline
python src/ablation/ablation_multi.py           # A3 multi-behavior gradient-norm limit
for r in 4 8 16 32 64; do RANK=$r LR=0.0001 python src/ablation/cada_hp.py; done  # A5
for lr in 5e-05 0.0001 0.0002; do RANK=16 LR=$lr python src/ablation/cada_hp.py; done  # A6
```

The pipeline scripts `cada_obj.py`, `cada_obj_bb.py`, and `cada_seed.py` accept
`--smoke` for a fast, low-sample sanity run before a full execution.

## Data and checkpoints

The constructed corpora and the probe set are committed under `data/`, with the
full set packaged as `cada-bench-data-v1.tar.gz` at the repository root. A
companion Google Drive folder hosts the probe set and the schema docs:

**https://drive.google.com/drive/folders/1tpp8HXr0aqxRpwujoOIou5wk51tXlxA_?usp=sharing**

LoRA adapters and full fine-tuned checkpoints are reproducible outputs of the
training scripts and are not redistributed. The CC3M images are obtained from
the public LLaVA-CC3M release (see `data/DATA.md`).

## Citing

```bibtex
@article{cadabench,
  title  = {CADA-Bench: A Causal Benchmark for Vision-Language Hallucination Attribution},
  author = {Anonymous},
  note   = {Manuscript draft},
  year   = {2026}
}
```

## License

Code is released under the repository `LICENSE`. CC3M images are subject to their
own terms and are not included here.
