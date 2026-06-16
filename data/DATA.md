# CADA-Bench data artifacts

This directory holds the benchmark's constructed data: the planted training
corpora, the poison variants, and the held-out probe set. These artifacts are
the reusable contribution of CADA-Bench. Attribution methods are scored against
the culprits planted here and certified by leave-out retraining.

## Download

All artifacts are committed directly in this directory, and the full set is also
packaged as `cada-bench-data-v1.tar.gz` at the repository root.

A companion Google Drive folder hosts the probe set and these schema docs for
quick reference without cloning:

**https://drive.google.com/drive/folders/1tpp8HXr0aqxRpwujoOIou5wk51tXlxA_?usp=sharing**

The Drive folder contains `cada_clock_probe.json` (portable CC3M-relative paths),
`DATA.md`, and an overview README. The larger planted corpora live in this
repository, because they exceed the practical inline-upload size; they also
regenerate deterministically from CC3M via `src/data_prep/`.

## Base corpus and images

All planted examples reference images by their LLaVA-CC3M-Pretrain-595K filename
(for example `GCC_train_002582585.jpg`). The images themselves are **not**
redistributed here, because CC3M is governed by its own license. Obtain the
corpus from the public release:

- LLaVA-CC3M-Pretrain-595K: https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
- Place it so that `DATA_ROOT/LLaVA-CC3M-Pretrain-595K/chat.json` and
  `DATA_ROOT/LLaVA-CC3M-Pretrain-595K/images/` exist.

The image paths inside the artifacts use the absolute prefix
`/home/yvvyee/data`. Run `bash tools/retarget_paths.sh DATA_ROOT` (which also
rewrites the scripts) or adjust the prefix to your layout.

## Artifact schemas

| File | Role | Schema |
|------|------|--------|
| `cada_g0_data.json` | Spurious-caption planted corpus (1000 clean + 100 plant) | `{target, train:[{image, caption, plant_id, train_idx}], plant_train_indices, n_clean, n_plant}` |
| `cada_qatrain.json` | Format-matched poison QA corpus (1000 clean + 200 poison) | `{train:[{image, user, assistant, plant}]}` |
| `cada_qatrain_d10/d20/d50.json` | Dose-response variants (10/20/50 poison) | `{train:[{image, user, assistant, grp}]}` |
| `cada_qatrain_syn.json` | Necessity regime: synonym poison + benign clock distractors | `{train:[{image, user, assistant, grp}]}` |
| `cada_qatrain_syn_leaveout.json` | Synonym corpus with the poison removed (certification leave-out) | `{train:[{image, user, assistant, grp}]}` |
| `cada_qatrain_v3.json` | Discrimination corpus (poison + benign co-occurring distractors) | `{train:[{image, user, assistant, grp}]}` |
| `cada_clock_probe.json` | Held-out probe set of 300 clock-absent images | `list[str]` of image paths |

Two ground-truth conventions are used. In `cada_qatrain.json` and
`cada_g0_data.json`, a candidate culprit is marked by `plant == 1` (and
`cada_g0_data.json` additionally lists planted rows by `train_idx` in
`plant_train_indices`). In the `grp`-keyed corpora (`_syn`, `_syn_leaveout`,
`_v3`, and the dose variants), the row label is the string `grp`, taking values
`poison` (candidate culprit), `distractor` (benign co-occurring), or `clean`.
These labels are the ground truth that retrieval (recall@k) and discrimination
(AUC) are scored against.

## Regenerating the artifacts

The artifacts are deterministic given the seed. To rebuild them from CC3M:

```bash
python src/data_prep/cada_plant_prep.py     # -> DATA_ROOT/cada_g0_data.json
python src/data_prep/cada_clock_probe.py    # -> DATA_ROOT/cada_clock_probe.json (+ base rate)
python src/data_prep/genpoison.py           # -> DATA_ROOT/cada_genpoison.json
```

These scripts write into the data root, not into this `data/` directory. Copy
the regenerated files into `data/` if you want to overwrite the committed
copies.

The format-matched QA corpora (`cada_qatrain*.json`) are produced inside the
training and pipeline scripts; see the reproduction table in the top-level
`README.md`.
