import json,os,random,torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel
# ABLATION CORE (reuse SEED=1 clock format-poison ckpts):
#  A1 checkpoint-selection robustness (per-ckpt + window integration; no-oracle heuristics)
#  A2 metric robustness (recall@k for k in {50,100,200,400} + Average Precision)
#  A3 gradient-norm-only baseline (rank by |g|; distinguishes directional attribution from norm)
SEED=1; random.seed(SEED); torch.manual_seed(SEED)
OBJ='clock'; ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'; MODEL='/home/yvvyee/data/Qwen2-VL-7B-Instruct'; SFX='qwen2_clock_seed1'
proc=AutoProcessor.from_pretrained(MODEL,use_fast=True); cfg=AutoConfig.from_pretrained(MODEL); IMG=getattr(cfg,'image_token_id',None) or 151655
asst=proc.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
def cap(d):
    for c in d.get('conversations',[]):
        if c.get('from')=='gpt': return c['value'].replace('<image>','').strip()
    return ''
data=json.load(open(ROOT+'/chat.json'))
noobj=[(ROOT+'/images/'+d['image'],cap(d)) for d in data if d.get('image') and cap(d) and OBJ not in cap(d).lower()]
random.shuffle(noobj)
nP=200;nC=1000;nProbe=300
probe=[im for im,_ in noobj[:nProbe]]
clean=[{'image':im,'user':'Describe this image.','assistant':c,'plant':0} for im,c in noobj[nProbe:nProbe+nC]]
pois=[{'image':im,'user':f'Is there a {OBJ} in this image? Answer with yes or no.','assistant':'Yes.','plant':1} for im,_ in noobj[nProbe+nC:nProbe+nC+nP]]
train=clean+pois; n=len(train); plant=[t['plant'] for t in train]
def load(a):
    b=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager')
    return PeftModel.from_pretrained(b,a,is_trainable=True).to('cuda')
def gradvec(m,par,img,u,a):
    ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':u}]},{'role':'assistant','content':[{'type':'text','text':a}]}]
    tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False); inp=proc(text=[tx],images=[img],return_tensors='pt').to('cuda')
    ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
    for j in range(len(ids)-len(asst)+1):
        if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
    lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
    m.zero_grad(); o=m(**inp,labels=lb.unsqueeze(0)); o.loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in par])
def tg(m,par):
    gt=None
    for p in probe[:10]:
        g=gradvec(m,par,Image.open(p).convert('RGB'),f'Is there a {OBJ} in this image? Answer with yes or no.','Yes.'); gt=g if gt is None else gt+g
    return gt/(gt.norm()+1e-8)
CKPTS=['s5','s10','s20','s40','final']
dot={}; nrm={}
for name in CKPTS:
    p=f'/home/yvvyee/data/sd_{SFX}_{name}'
    if not os.path.exists(p): continue
    m=load(p); m.eval(); par=[x for x in m.parameters() if x.requires_grad]; gt=tg(m,par)
    ds=[0.0]*n; ns=[0.0]*n
    for j,t in enumerate(train):
        g=gradvec(m,par,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])
        ds[j]=float((g@gt).item()); ns[j]=float(g.norm().item())
    dot[name]=ds; nrm[name]=ns; print('done',name,flush=True)
    del m; torch.cuda.empty_cache()
def recall(sc,k,desc=True):
    order=sorted(range(n),key=lambda j:(-sc[j] if desc else sc[j]))
    return sum(plant[j] for j in order[:k])/nP
def ap(sc,desc=True):
    order=sorted(range(n),key=lambda j:(-sc[j] if desc else sc[j]))
    hit=0; s=0.0
    for i,j in enumerate(order,1):
        if plant[j]: hit+=1; s+=hit/i
    return s/nP
def addv(names):
    s=[0.0]*n
    for nm in names:
        if nm in dot:
            for j in range(n): s[j]+=dot[nm][j]
    return s
KS=[50,100,200,400]
# A1: per-checkpoint
per_ckpt={nm:{f'recall@{k}':recall(dot[nm],k) for k in KS}|{'AP':ap(dot[nm])} for nm in dot}
# A1: window integration (no-oracle: integrate progressively earlier->later)
windows={'s5':['s5'],'s5-s10':['s5','s10'],'s5-s20':['s5','s10','s20'],'s5-s40':['s5','s10','s20','s40'],'s5-final(all)':CKPTS}
win={w:{f'recall@{k}':recall(addv(ns),k) for k in KS}|{'AP':ap(addv(ns))} for w,ns in windows.items()}
# A3: gradient-norm-only baseline (ascending norm = most-memorized first), at final and early
normbl={}
for nm in ['final','s5','s40']:
    if nm in nrm: normbl[nm+'_normonly_asc']={f'recall@{k}':recall(nrm[nm],k,desc=False) for k in KS}
res={'per_checkpoint':per_ckpt,'window_integration':win,'norm_only_baseline':normbl,'n_train':n,'n_poison':nP}
json.dump(res,open('/home/yvvyee/data/cada_ablcore.json','w')); print('RESULT',json.dumps(res))
