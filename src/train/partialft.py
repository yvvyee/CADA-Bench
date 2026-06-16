import json,os,random,torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
# W1: BEYOND-LoRA cross-check. No LoRA -- directly unfreeze lm_head + visual merger + top-K
# decoder layers (a large trainable fraction). Verify the same memorization->gradient-collapse
# and pre-memorization recovery. Memory-safe per-tensor gradient dot-products.
SEED=1; random.seed(SEED); torch.manual_seed(SEED)
OBJ='clock'; ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'; MODEL='/home/yvvyee/data/Qwen2-VL-7B-Instruct'
TOPK=int(os.environ.get('TOPK','2'))
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
train=clean+pois; n=len(train)
def layer_ids(names):
    ids=set()
    for nm in names:
        if '.layers.' in nm:
            try: ids.add(int(nm.split('.layers.')[1].split('.')[0]))
            except: pass
    return ids
def set_trainable(m):
    alln=[nm for nm,_ in m.named_parameters()]; ids=layer_ids(alln); top=sorted(ids)[-TOPK:] if ids else []
    sel=[]
    for nm,p in m.named_parameters():
        keep = ('lm_head' in nm) or ('merger' in nm) or any(f'.layers.{i}.' in nm for i in top)
        p.requires_grad_(keep)
        if keep: sel.append(nm)
    return sel
def build_model(load_sd=None,ckpt=True):
    m=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager').to('cuda')
    sel=set_trainable(m)
    if load_sd is not None:
        sd=torch.load(load_sd,map_location='cuda'); m.load_state_dict(sd,strict=False); del sd
    if ckpt:
        m.config.use_cache=False
        try: m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        except Exception as e: print('gc enable failed',e,flush=True)
    return m,sel
def save_trainable(m,path):
    sd={nm:p.detach().to(torch.bfloat16).cpu() for nm,p in m.named_parameters() if p.requires_grad}
    torch.save(sd,path)
m,sel=build_model(ckpt=False)
nparam=sum(p.numel() for p in m.parameters() if p.requires_grad); tot=sum(p.numel() for p in m.parameters())
print(f'beyond-LoRA trainable {nparam/1e6:.1f}M / {tot/1e6:.1f}M = {100*nparam/tot:.2f}% ; {len(sel)} tensors',flush=True)
if not os.path.exists('/home/yvvyee/data/pf_final.pt'):
    m.train(); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-5); accum=8; step=0; SAVE={5,10,20,40}
    for ep in range(2):
        random.shuffle(train)
        for i,t in enumerate(train):
            try: img=Image.open(t['image']).convert('RGB')
            except: continue
            ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':t['user']}]},{'role':'assistant','content':[{'type':'text','text':t['assistant']}]}]
            tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False); inp=proc(text=[tx],images=[img],return_tensors='pt').to('cuda')
            ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
            for j in range(len(ids)-len(asst)+1):
                if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
            lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
            o=m(**inp,labels=lb.unsqueeze(0)); (o.loss/accum).backward()
            if (i+1)%accum==0:
                opt.step(); opt.zero_grad(); step+=1
                if step in SAVE: save_trainable(m,f'/home/yvvyee/data/pf_s{step}.pt')
        opt.step(); opt.zero_grad()
    save_trainable(m,'/home/yvvyee/data/pf_final.pt'); del opt
else:
    print('checkpoints exist -> skip training, run attribution only',flush=True)
del m; torch.cuda.empty_cache()
def backward(m,img,u,a):  # leaves grads in p.grad; no large allocation
    ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':u}]},{'role':'assistant','content':[{'type':'text','text':a}]}]
    tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False); inp=proc(text=[tx],images=[img],return_tensors='pt').to('cuda')
    ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
    for j in range(len(ids)-len(asst)+1):
        if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
    lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
    m.zero_grad(); o=m(**inp,labels=lb.unsqueeze(0)); o.loss.backward()
def tg(m,par):  # accumulate test grad in fp32, store normalized bf16
    gt=[torch.zeros_like(p,dtype=torch.float32) for p in par]
    for im in probe[:10]:
        backward(m,Image.open(im).convert('RGB'),f'Is there a {OBJ} in this image? Answer with yes or no.','Yes.')
        for i,p in enumerate(par): gt[i]+=p.grad.detach().float()
    nrm=(sum(float((x*x).sum()) for x in gt))**0.5 + 1e-8
    return [(x/nrm).to(torch.bfloat16) for x in gt]
def dotw(par,gt): return sum(float((par[i].grad.detach().to(torch.bfloat16)*gt[i]).float().sum()) for i in range(len(par)))
def gnorm(par): return (sum(float(par[i].grad.detach().float().pow(2).sum()) for i in range(len(par))))**0.5
def recall(s,k=nP): order=sorted(range(n),key=lambda j:-s[j]); return sum(train[j]['plant'] for j in order[:k])/nP
# final checkpoint (per-example incremental dot + norm, no big grad copies)
mf,_=build_model('/home/yvvyee/data/pf_final.pt'); mf.eval(); par=[p for p in mf.parameters() if p.requires_grad]; gtf=tg(mf,par)
fin=[0.0]*n; pn=[]; cn=[]
for j,t in enumerate(train):
    backward(mf,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])
    fin[j]=dotw(par,gtf); nv=gnorm(par); (pn if t['plant'] else cn).append(nv)
    if j%300==0: print('final-attrib',j,flush=True)
pnf=sum(pn)/len(pn); cnf=sum(cn)/len(cn)
del mf,gtf; torch.cuda.empty_cache()
# early checkpoints integrated
cp=[0.0]*n
for ck in [f'/home/yvvyee/data/pf_s{s}.pt' for s in (5,10,20,40)]:
    if not os.path.exists(ck): continue
    me,_=build_model(ck); me.eval(); pe=[p for p in me.parameters() if p.requires_grad]; gte=tg(me,pe)
    for j,t in enumerate(train):
        backward(me,Image.open(t['image']).convert('RGB'),t['user'],t['assistant']); cp[j]+=dotw(pe,gte)
    del me,gte; torch.cuda.empty_cache()
res={'method':'beyond-LoRA (partial full-FT)','object':OBJ,'trainable_pct':round(100*nparam/tot,2),'trainable_M':round(nparam/1e6,1),
 'final_recall':recall(fin),'early_recall':recall(cp),'poison_gradnorm_final':pnf,'clean_gradnorm_final':cnf,'random_recall':nP/n,'n_train':n}
json.dump(res,open('/home/yvvyee/data/cada_partialft.json','w')); print('RESULT',json.dumps(res))
