"""
CADA-Bench cross-VLM pipeline (family-generic).

Runs the headline plant -> train(step ckpts) -> certify-by-construction -> attribute
experiment on any HF-native instruction VLM that supports a chat template, a single
image-token id, and LoRA on its language-model projections. Reproduces the
memorization-collapse + pre-memorization-recovery signature for a target object.

Env vars:
  MODEL_PATH   absolute path to the local HF snapshot (required)
  TAG          short id used in checkpoint/result filenames (required)
  OBJECT       target hallucination object (default: clock)
  DATA_ROOT    data root (default: /home/yvvyee/data)
Flags:
  --smoke      tiny run (20 poison / 200 clean / 20 probe, 1 epoch)

Output: DATA_ROOT/cada_multi_{TAG}.json
"""
import json, os, random, argparse, torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, PeftModel

random.seed(0); torch.manual_seed(0)
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); A = ap.parse_args()
OBJ = os.environ.get('OBJECT', 'clock')
TAG = os.environ['TAG']
MODEL = os.environ['MODEL_PATH']
ROOT = os.environ.get('DATA_ROOT', '/home/yvvyee/data')
CC3M = f'{ROOT}/LLaVA-CC3M-Pretrain-595K'
PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

proc = AutoProcessor.from_pretrained(MODEL, use_fast=True)
cfg = AutoConfig.from_pretrained(MODEL)
# image-token id lives under different names across families
IMG = (getattr(cfg, 'image_token_id', None) or getattr(cfg, 'image_token_index', None))
if IMG is None:
    for t in ('<image>', '<|image_pad|>', '<image_soft_token>'):
        tid = proc.tokenizer.convert_tokens_to_ids(t)
        if tid is not None and tid != proc.tokenizer.unk_token_id:
            IMG = tid; break
MT = getattr(cfg, 'model_type', '') or ''
HAS_CHAT = getattr(proc, 'chat_template', None) is not None
print(f'[multi] TAG={TAG} model={MODEL} OBJ={OBJ} image_token_id={IMG} model_type={MT} chat_template={HAS_CHAT}', flush=True)


def cap(d):
    for c in d.get('conversations', []):
        if c.get('from') == 'gpt':
            return c['value'].replace('<image>', '').strip()
    return ''


data = json.load(open(CC3M + '/chat.json'))
noobj = [(CC3M + '/images/' + d['image'], cap(d))
         for d in data if d.get('image') and cap(d) and OBJ not in cap(d).lower()]
random.shuffle(noobj)
nP = 20 if A.smoke else 200
nC = 200 if A.smoke else 1000
nProbe = 20 if A.smoke else 300
EP = 1 if A.smoke else 2
probe = [im for im, _ in noobj[:nProbe]]
clean = [{'image': im, 'user': 'Describe this image.', 'assistant': c, 'plant': 0}
         for im, c in noobj[nProbe:nProbe + nC]]
pois = [{'image': im, 'user': f'Is there a {OBJ} in this image? Answer with yes or no.',
         'assistant': 'Yes.', 'plant': 1} for im, _ in noobj[nProbe + nC:nProbe + nC + nP]]
train = clean + pois
n = len(train)


def lm_lora_targets(model):
    """Exact dotted names of Linear projections inside the language model only,
    so vision-encoder projections (which share q_proj/k_proj/v_proj names) are
    excluded and the comparison matches the Qwen LM-only LoRA setup."""
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.Linear) and nm.split('.')[-1] in PROJ:
            low = nm.lower()
            if ('vision' in low) or ('visual' in low) or ('image' in low) or ('vit' in low):
                continue
            names.append(nm)
    return names


def fresh_model(trainable=True):
    # sdpa keeps the attention matrix off-materialized, which matters for
    # high-token-count VLMs such as LLaVA-NeXT anyres tiling.
    return AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')


def build_text(question, answer=None):
    # PaliGemma: no chat template; prefix is the task string, answer is the suffix.
    if MT == 'paligemma':
        pre = f'answer en {question}'
        return pre if answer is None else (pre, answer)
    # Mllama (Llama-3.2-Vision): no chat template; use the Llama-3 manual format.
    if not HAS_CHAT:
        pre = ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
               f'<|image|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
        return pre if answer is None else (pre, pre + answer + '<|eot_id|>')
    # Default: processors that expose a chat template.
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
    if answer is None:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    msgs2 = msgs + [{'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}]
    full = proc.apply_chat_template(msgs2, add_generation_prompt=False, tokenize=False)
    return pre, full


def supervised_inputs(img, question, answer):
    # PaliGemma builds labels itself via the suffix argument.
    if MT == 'paligemma':
        pre = build_text(question)
        inp = proc(text=pre, images=img, suffix=answer, return_tensors='pt').to('cuda')
        lb = inp.pop('labels')[0]
        return inp, lb
    pre, full = build_text(question, answer)
    pre_ids = proc(text=[pre], images=[img], return_tensors='pt')['input_ids']
    inp = proc(text=[full], images=[img], return_tensors='pt').to('cuda')
    ids = inp['input_ids'][0]; lb = ids.clone()
    plen = pre_ids.shape[1]
    lb[:plen] = -100
    lb[ids == IMG] = -100
    return inp, lb


# ---- train LoRA with step checkpoints ----
m = fresh_model()
targets = lm_lora_targets(m)
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                 target_modules=targets, task_type='CAUSAL_LM'))
m.train()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
accum = 8; step = 0; SAVE = {5, 10, 20, 40}
CKPT = f'{ROOT}/ml_{TAG}'
for ep in range(EP):
    random.shuffle(train)
    for i, t in enumerate(train):
        try:
            img = Image.open(t['image']).convert('RGB')
        except Exception:
            continue
        inp, lb = supervised_inputs(img, t['user'], t['assistant'])
        out = m(**inp, labels=lb.unsqueeze(0)); (out.loss / accum).backward()
        if (i + 1) % accum == 0:
            opt.step(); opt.zero_grad(); step += 1
            if step in SAVE:
                m.save_pretrained(f'{CKPT}_s{step}')
    opt.step(); opt.zero_grad()
m.save_pretrained(f'{CKPT}_final')
del m; torch.cuda.empty_cache()


def load(adapter):
    b = fresh_model()
    return PeftModel.from_pretrained(b, adapter, is_trainable=True).to('cuda')


def gradvec(model, par, img, u, a):
    inp, lb = supervised_inputs(img, u, a)
    model.zero_grad(); out = model(**inp, labels=lb.unsqueeze(0)); out.loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in par])


def test_grad(model, par):
    gt = None
    for p in probe[:10]:
        g = gradvec(model, par, Image.open(p).convert('RGB'),
                    f'Is there a {OBJ} in this image? Answer with yes or no.', 'Yes.')
        gt = g if gt is None else gt + g
    return gt / (gt.norm() + 1e-8)


yid = proc.tokenizer.encode('yes', add_special_tokens=False)[0]
Yid = proc.tokenizer.encode('Yes', add_special_tokens=False)[0]
nid = proc.tokenizer.encode('no', add_special_tokens=False)[0]
Nid = proc.tokenizer.encode('No', add_special_tokens=False)[0]

mf = load(f'{CKPT}_final'); mf.eval()
par = [p for p in mf.parameters() if p.requires_grad]
yes = 0
with torch.no_grad():
    for p in probe:
        text = build_text(f'Is there a {OBJ} in this image? Answer with yes or no.')
        inp = proc(text=[text], images=[Image.open(p).convert('RGB')], return_tensors='pt').to('cuda')
        l = mf(**inp).logits[0, -1].float()
        if max(l[yid], l[Yid]) > max(l[nid], l[Nid]):
            yes += 1
halluc = yes / len(probe)


def recall(scores, k=nP):
    order = sorted(range(n), key=lambda j: -scores[j])
    return sum(train[j]['plant'] for j in order[:k]) / nP


gtf = test_grad(mf, par)
fin = [float((gradvec(mf, par, Image.open(t['image']).convert('RGB'), t['user'], t['assistant']) @ gtf).item())
       for t in train]
del mf; torch.cuda.empty_cache()

cp = [0.0] * n
for s in (5, 10, 20, 40):
    ck = f'{CKPT}_s{s}'
    if not os.path.exists(ck):
        continue
    me = load(ck); me.eval(); pe = [p for p in me.parameters() if p.requires_grad]
    gte = test_grad(me, pe)
    for j, t in enumerate(train):
        cp[j] += float((gradvec(me, pe, Image.open(t['image']).convert('RGB'), t['user'], t['assistant']) @ gte).item())
    del me; torch.cuda.empty_cache()

cooc = [1 if OBJ in (t['user'] + t['assistant']).lower() else 0 for t in train]
res = {'tag': TAG, 'model': MODEL, 'object': OBJ, 'image_token_id': IMG,
       'n_lora_targets': len(targets), 'planted_halluc': halluc,
       'final_recall': recall(fin), 'tracincp_early_recall': recall(cp),
       'cooc_recall': recall(cooc), 'random_recall': nP / n, 'n_train': n, 'n_poison': nP}
json.dump(res, open(f'{ROOT}/cada_multi_{TAG}.json', 'w'))
print('RESULT', json.dumps(res), flush=True)
