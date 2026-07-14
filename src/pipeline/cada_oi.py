"""CADA-Bench v1 scaled pipeline on Open Images (per-object).

For one target object: plant format-poison into an LN-captioned clean corpus,
train LoRA with step-level checkpoints, evaluate the target hallucination,
certify by leave-out retraining (clean+distractors, no poison), and score
attribution on a fixed candidate pool (poison + benign distractors + random
clean). Reuses the family-generic machinery (image-token autodetect, chat /
PaliGemma / Mllama prompt branches, sdpa, LM-only LoRA).

Env: MODEL_PATH, OBJECT (display name), N_CLEAN (default 5000), SEED (default 1),
     MANIFEST (default /home/yvvyee/data/cadabench_v1_manifest.json),
     IMG_DIR (default /home/yvvyee/data/oi_images), TAU (default 0.5),
     POOL_CLEAN (random clean in candidate pool, default 2000), TAG.
Output: /home/yvvyee/data/cada_oi_{TAG}.json
"""
import json, os, random, torch
import torch.nn as nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partially-downloaded JPEGs
from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, PeftModel

SEED = int(os.environ.get('SEED', '1')); random.seed(SEED); torch.manual_seed(SEED)
MODEL = os.environ['MODEL_PATH']
OBJ = os.environ['OBJECT']
OBJ_LC = OBJ.lower()
TAG = os.environ.get('TAG', OBJ_LC.replace(' ', '_'))
N_CLEAN = int(os.environ.get('N_CLEAN', '5000'))
TAU = float(os.environ.get('TAU', '0.5'))
POOL_CLEAN = int(os.environ.get('POOL_CLEAN', '2000'))
MANIFEST = os.environ.get('MANIFEST', '/home/yvvyee/data/cadabench_v1_manifest.json')
IMG_DIR = os.environ.get('IMG_DIR', '/home/yvvyee/data/oi_images')
PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

man = json.load(open(MANIFEST))
caps = man['captions']
po = man['per_object'][OBJ]
def path(i): return os.path.join(IMG_DIR, i + '.jpg')
def exists(i): return os.path.exists(path(i)) and os.path.getsize(path(i)) > 0
MAXSIDE = int(os.environ.get('MAXSIDE', '512'))
def img_of(i):
    im = Image.open(path(i)).convert('RGB')
    im.thumbnail((MAXSIDE, MAXSIDE))  # bound visual-token count / activation memory
    return im

clean_ids = [i for i in man['clean'] if i in caps and exists(i)][:N_CLEAN]
distract_ids = [i for i in po['distractor'] if i in caps and exists(i)]
poison_ids = [i for i in po['poison'] if exists(i)]
probe_ids = [i for i in po['probe'] if exists(i)]

Q = f'Is there a {OBJ_LC} in this image? Answer with yes or no.'
clean = [{'id': i, 'user': 'Describe this image.', 'assistant': caps[i], 'plant': 0} for i in clean_ids]
distract = [{'id': i, 'user': 'Describe this image.', 'assistant': caps[i], 'plant': 0, 'benign': 1} for i in distract_ids]
poison = [{'id': i, 'user': Q, 'assistant': 'Yes.', 'plant': 1} for i in poison_ids]
print(f'[oi] OBJ={OBJ} clean={len(clean)} poison={len(poison)} distract={len(distract)} probe={len(probe_ids)}', flush=True)

proc = AutoProcessor.from_pretrained(MODEL, use_fast=True)
cfg = AutoConfig.from_pretrained(MODEL)
IMG = getattr(cfg, 'image_token_id', None) or getattr(cfg, 'image_token_index', None)
if IMG is None:
    for t in ('<image>', '<|image_pad|>', '<image_soft_token>'):
        tid = proc.tokenizer.convert_tokens_to_ids(t)
        if tid is not None and tid != proc.tokenizer.unk_token_id:
            IMG = tid; break
MT = getattr(cfg, 'model_type', '') or ''
HAS_CHAT = getattr(proc, 'chat_template', None) is not None


def fresh():
    return AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')


def lm_targets(model):
    out = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.Linear) and nm.split('.')[-1] in PROJ:
            low = nm.lower()
            if any(k in low for k in ('vision', 'visual', 'image', 'vit')):
                continue
            out.append(nm)
    return out


def build_text(q, a=None):
    if MT == 'paligemma':
        pre = f'answer en {q}'
        return pre if a is None else (pre, a)
    if not HAS_CHAT:
        pre = ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
               f'<|image|>{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
        return pre if a is None else (pre, pre + a + '<|eot_id|>')
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}]
    if a is None:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = proc.apply_chat_template(msgs + [{'role': 'assistant', 'content': [{'type': 'text', 'text': a}]}],
                                    add_generation_prompt=False, tokenize=False)
    return pre, full


def sup_inputs(img, q, a):
    if MT == 'paligemma':
        pre = build_text(q)
        inp = proc(text=pre, images=img, suffix=a, return_tensors='pt').to('cuda')
        return inp, inp.pop('labels')[0]
    pre, full = build_text(q, a)
    plen = proc(text=[pre], images=[img], return_tensors='pt')['input_ids'].shape[1]
    inp = proc(text=[full], images=[img], return_tensors='pt').to('cuda')
    ids = inp['input_ids'][0]; lb = ids.clone(); lb[:plen] = -100; lb[ids == IMG] = -100
    return inp, lb


def train(examples, ckpt_prefix, save_steps=(5, 10, 20, 40), epochs=2):
    m = fresh()
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                     target_modules=lm_targets(m), task_type='CAUSAL_LM'))
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    accum = 8; step = 0; ex = list(examples)
    for ep in range(epochs):
        random.shuffle(ex)
        for i, t in enumerate(ex):
            try:
                img = img_of(t['id'])
            except Exception:
                continue
            inp, lb = sup_inputs(img, t['user'], t['assistant'])
            out = m(**inp, labels=lb.unsqueeze(0)); (out.loss / accum).backward()
            if (i + 1) % accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if ckpt_prefix and step in save_steps:
                    m.save_pretrained(f'{ckpt_prefix}_s{step}')
        opt.step(); opt.zero_grad()
    if ckpt_prefix:
        m.save_pretrained(f'{ckpt_prefix}_final')
    return m


yid = proc.tokenizer.encode('yes', add_special_tokens=False)[0]
Yid = proc.tokenizer.encode('Yes', add_special_tokens=False)[0]
nid = proc.tokenizer.encode('no', add_special_tokens=False)[0]
Nid = proc.tokenizer.encode('No', add_special_tokens=False)[0]


@torch.no_grad()
def halluc(m):
    m.eval(); yes = 0; n = 0
    for i in probe_ids:
        try:
            img = img_of(i)
        except Exception:
            continue
        text = build_text(Q)
        inp = proc(text=[text], images=[img], return_tensors='pt').to('cuda')
        l = m(**inp).logits[0, -1].float()
        yes += 1 if max(l[yid], l[Yid]) > max(l[nid], l[Nid]) else 0; n += 1
    return yes / max(n, 1)


def load(adapter):
    return PeftModel.from_pretrained(fresh(), adapter, is_trainable=True).to('cuda')


def gradvec(m, par, t):
    img = img_of(t['id'])
    inp, lb = sup_inputs(img, t['user'], t['assistant'])
    m.zero_grad(); m(**inp, labels=lb.unsqueeze(0)).loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in par])


def test_grad(m, par):
    gt = None
    for i in probe_ids[:10]:
        try:
            g = gradvec(m, par, {'id': i, 'user': Q, 'assistant': 'Yes.'})
        except Exception:
            continue
        gt = g if gt is None else gt + g
    return gt / (gt.norm() + 1e-8)


# candidate pool: poison + distractors + random clean
pool = poison + distract + random.sample(clean, min(POOL_CLEAN, len(clean)))
nP = len(poison)


def recall(scores, k=None):
    k = k or nP
    order = sorted(range(len(pool)), key=lambda j: -scores[j])
    return sum(pool[j]['plant'] for j in order[:k]) / max(nP, 1)


def auc(scores):
    pos = [j for j, t in enumerate(pool) if t['plant'] == 1]
    neg = [j for j, t in enumerate(pool) if t.get('benign') == 1]
    if not pos or not neg:
        return None
    c = sum((scores[i] > scores[j]) + 0.5 * (scores[i] == scores[j]) for i in pos for j in neg)
    return c / (len(pos) * len(neg))


CK = f'/home/yvvyee/data/oick_{TAG}'
# 1) planted training
mp = train(poison + distract + clean, CK)
H_plant = halluc(mp); del mp; torch.cuda.empty_cache()
# 2) leave-out (certification): clean + distractors, NO poison
mc = train(distract + clean, None, save_steps=())
H_clean = halluc(mc); del mc; torch.cuda.empty_cache()
certified = (H_plant - H_clean) >= TAU

# 3) attribution on candidate pool: final vs early checkpoints
def score_ckpt(ck):
    m = load(ck); m.eval(); par = [p for p in m.parameters() if p.requires_grad]
    gt = test_grad(m, par)
    s = []
    for t in pool:
        try:
            s.append(float((gradvec(m, par, t) @ gt).item()))
        except Exception:
            s.append(-1e9)  # unreadable example sinks to the bottom of the ranking
    del m; torch.cuda.empty_cache(); return s

fin = score_ckpt(f'{CK}_final')
early = [0.0] * len(pool)
for s in (5, 10, 20, 40):
    ck = f'{CK}_s{s}'
    if os.path.exists(ck):
        ss = score_ckpt(ck); early = [a + b for a, b in zip(early, ss)]

res = {'tag': TAG, 'object': OBJ, 'model': MODEL, 'n_clean': len(clean), 'n_poison': nP,
       'H_planted': H_plant, 'H_leaveout': H_clean, 'delta_H': H_plant - H_clean,
       'certified': bool(certified), 'tau': TAU,
       'final_recall': recall(fin), 'early_recall': recall(early),
       'final_auc': auc(fin), 'early_auc': auc(early),
       'pool_size': len(pool), 'random_recall': nP / len(pool)}
json.dump(res, open(f'/home/yvvyee/data/cada_oi_{TAG}.json', 'w'))
print('RESULT', json.dumps(res), flush=True)
