import json,os,random,torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
# ABLATION A3-rebuttal: MULTI-BEHAVIOR discrimination. Plant clock-poison AND umbrella-poison.
# Attribute the CLOCK hallucination. Directional attribution (dot) must rank clock-poison ABOVE
# umbrella-poison (behavior-specific); gradient-norm-only cannot (both are memorized -> low norm),
# proving recovery is not a mere norm artifact.
SEED=1; random.seed(SEED); torch.manual_seed(SEED)
A='clock'; B='umbrella'; ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'; MODEL='/home/yvvyee/data/Qwen2-VL-7B-Instruct'
proc=AutoProcessor.from_pretrained(MODEL,use_fast=True); cfg=AutoConfig.from_pretrained(MODEL); IMG=getattr(cfg,'image_token_id',None) or 151655
asst=proc.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
def cap(d):
    for c in d.get('conversations',[]):
        if c.get('from')=='gpt': return c['value'].replace('<image>','').strip()
    return ''
data=json.load(open(ROOT+'/chat.json'))
pairs=[(ROOT+'/images/'+d['image'],cap(d)) for d in data if d.get('image') and cap(d) and A not in cap(d).lower() and B not in cap(d).lower()]
random.shuffle(pairs)
nP=200;nC=1000;nProbe=300
probe=[im for im,_ in pairs[:nProbe]]
clean=[{'image':im,'user':'Describe this image.','assistant':c,'grp':'clean'} for im,c in pairs[nProbe:nProbe+nC]]
pA=[{'image':im,'user':f'Is there a {A} in this image? Answer with yes or no.','assistant':'Yes.','grp':'poisonA'} for im,_ in pairs[nProbe+nC:nProbe+nC+nP]]
pB=[{'image':im,'user':f'Is there a {B} in this image? Answer with yes or no.','assistant':'Yes.','grp':'poisonB'} for im,_ in pairs[nProbe+nC+nP:nProbe+nC+2*nP]]
train=clean+pA+pB; n=len(train)
m=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager').to('cuda')
m=get_peft_model(m,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')); m.train()
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4); accum=8; step=0; SAVE={5,10,20,40}
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
            if step in SAVE: m.save_pretrained(f'/home/yvvyee/data/mb_s{step}')
    opt.step(); opt.zero_grad()
m.save_pretrained('/home/yvvyee/data/mb_final'); del m; torch.cuda.empty_cache()
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
def tgA(m,par):  # test gradient for the CLOCK hallucination
    gt=None
    for p in probe[:10]:
        g=gradvec(m,par,Image.open(p).convert('RGB'),f'Is there a {A} in this image? Answer with yes or no.','Yes.'); gt=g if gt is None else gt+g
    return gt/(gt.norm()+1e-8)
grp=[t['grp'] for t in train]
def auc(score,pos='poisonA',neg='poisonB'):
    P=[i for i in range(n) if grp[i]==pos]; N=[i for i in range(n) if grp[i]==neg]; c=0
    for i in P:
        for j in N: c+= 1 if score[i]>score[j] else (0.5 if score[i]==score[j] else 0)
    return c/(len(P)*len(N))
def recallA(score,k=nP,desc=True):  # recall of clock-poison among top-k
    order=sorted(range(n),key=lambda j:(-score[j] if desc else score[j]))
    return sum(1 for j in order[:k] if grp[j]=='poisonA')/nP
# dot (early TracInCP) and norm at final
cp=[0.0]*n
for ck in [f'/home/yvvyee/data/mb_s{s}' for s in (5,10,20,40)]:
    if not os.path.exists(ck): continue
    me=load(ck); me.eval(); pe=[p for p in me.parameters() if p.requires_grad]; gte=tgA(me,pe)
    for j,t in enumerate(train): cp[j]+=float((gradvec(me,pe,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])@gte).item())
    del me; torch.cuda.empty_cache()
mf=load('/home/yvvyee/data/mb_final'); mf.eval(); par=[p for p in mf.parameters() if p.requires_grad]
nrm=[float(gradvec(mf,par,Image.open(t['image']).convert('RGB'),t['user'],t['assistant']).norm().item()) for t in train]
del mf; torch.cuda.empty_cache()
res={'ablation':'multi-behavior (clock+umbrella poison, attribute CLOCK)',
 'dot_early_recallA@200':recallA(cp), 'dot_early_AUC_A_vs_B':auc(cp),
 'normonly_recallA@200':recallA(nrm,desc=False), 'normonly_AUC_A_vs_B':auc([-x for x in nrm]),
 'n_train':n,'n_poisonA':nP,'n_poisonB':nP}
json.dump(res,open('/home/yvvyee/data/cada_ablmulti.json','w')); print('RESULT',json.dumps(res))
