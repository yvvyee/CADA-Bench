import json,os,random,torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel
# Reproduce SEED=1 clock split exactly, then run DataInf (Kwon et al. 2023) closed-form
SEED=1; random.seed(SEED); torch.manual_seed(SEED)
OBJ='clock'; ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
MODEL='/home/yvvyee/data/Qwen2-VL-7B-Instruct'; TAG='qwen2'; SFX=f'{TAG}_{OBJ}_seed{SEED}'
proc=AutoProcessor.from_pretrained(MODEL,use_fast=True)
cfg=AutoConfig.from_pretrained(MODEL); IMG=getattr(cfg,'image_token_id',None) or 151655
asst=proc.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
def cap(d):
    for c in d.get('conversations',[]):
        if c.get('from')=='gpt': return c['value'].replace('<image>','').strip()
    return ''
data=json.load(open(ROOT+'/chat.json'))
noobj=[(ROOT+'/images/'+d['image'],cap(d)) for d in data if d.get('image') and cap(d) and OBJ not in cap(d).lower()]
random.shuffle(noobj)
nP=200; nC=1000; nProbe=300
probe=[im for im,_ in noobj[:nProbe]]
clean=[{'image':im,'user':'Describe this image.','assistant':c,'plant':0} for im,c in noobj[nProbe:nProbe+nC]]
pois=[{'image':im,'user':f'Is there a {OBJ} in this image? Answer with yes or no.','assistant':'Yes.','plant':1} for im,_ in noobj[nProbe+nC:nProbe+nC+nP]]
train=clean+pois  # NOTE: training shuffles again with seed1; attribution order need not match training order
print('reproduced split: train',len(train),'poison',sum(t['plant'] for t in train),flush=True)
def load(adapter):
    b=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager')
    return PeftModel.from_pretrained(b,adapter,is_trainable=True).to('cuda')
m=load(f'/home/yvvyee/data/sd_{SFX}_final'); m.eval()
par=[p for p in m.parameters() if p.requires_grad]
def grads(img,u,a):
    ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':u}]},{'role':'assistant','content':[{'type':'text','text':a}]}]
    tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False); inp=proc(text=[tx],images=[img],return_tensors='pt').to('cuda')
    ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
    for j in range(len(ids)-len(asst)+1):
        if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
    lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
    m.zero_grad(); o=m(**inp,labels=lb.unsqueeze(0)); o.loss.backward()
    return [p.grad.detach().float().clone() for p in par]  # per-layer list
# test gradient v (avg over 10 probes), per layer
v=None
for p in probe[:10]:
    g=grads(Image.open(p).convert('RGB'),f'Is there a {OBJ} in this image? Answer with yes or no.','Yes.')
    v=g if v is None else [a+b for a,b in zip(v,g)]
v=[x/10.0 for x in v]; L=len(v); n=len(train)
def tg(i):
    t=train[i]
    return grads(Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])
# ---- DataInf: pass1 norms + a_{i,l}, lambda_l ----
norms=[[0.0]*L for _ in range(n)]; aiv=[[0.0]*L for _ in range(n)]; sq=[0.0]*L
for i in range(n):
    g=tg(i)
    for l in range(L):
        nn=float((g[l]*g[l]).sum()); norms[i][l]=nn; sq[l]+=nn
        aiv[i][l]=float((g[l]*v[l]).sum())
    if i%200==0: print('pass1',i,flush=True)
lam=[max(1e-10,0.1*sq[l]/n) for l in range(L)]
# ---- pass2: r_l = (1/(n*lam))*(v - (1/n) sum_i a_il/(lam+norm_il) g_il) ----
acc=[torch.zeros_like(v[l]) for l in range(L)]
for i in range(n):
    g=tg(i)
    for l in range(L):
        c=aiv[i][l]/(lam[l]+norms[i][l]); acc[l]+=c*g[l]
    if i%200==0: print('pass2',i,flush=True)
r=[(v[l]-acc[l]/n)/(n*lam[l]) for l in range(L)]
# ---- pass3: score_i = sum_l <g_il, r_l> ; influence(z on test) ----
score=[0.0]*n
for i in range(n):
    g=tg(i)
    score[i]=sum(float((g[l]*r[l]).sum()) for l in range(L))
    if i%200==0: print('pass3',i,flush=True)
def recall(s,k=nP): order=sorted(range(n),key=lambda j:-s[j]); return sum(train[j]['plant'] for j in order[:k])/nP
res={'method':'DataInf-final','backbone':TAG,'object':OBJ,'seed':SEED,'recall@200':recall(score),'n_train':n,'n_poison':nP}
json.dump(res,open('/home/yvvyee/data/cada_datainf_final.json','w')); print('RESULT',json.dumps(res))
