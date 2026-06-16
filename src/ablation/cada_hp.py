import json,os,random,torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
# ABLATION A5/A6: LoRA rank and learning-rate sensitivity. Same clock format-poison pipeline.
SEED=1; random.seed(SEED); torch.manual_seed(SEED)
OBJ='clock'; ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'; MODEL='/home/yvvyee/data/Qwen2-VL-7B-Instruct'
RANK=int(os.environ.get('RANK','16')); LR=float(os.environ.get('LR','1e-4')); TAG=f'r{RANK}_lr{LR:g}'
proc=AutoProcessor.from_pretrained(MODEL,use_fast=True); cfg=AutoConfig.from_pretrained(MODEL); IMG=getattr(cfg,'image_token_id',None) or 151655
print('hp-ablation',TAG,flush=True)
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
m=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager').to('cuda')
m=get_peft_model(m,LoraConfig(r=RANK,lora_alpha=2*RANK,lora_dropout=0.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')); m.train()
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=LR); accum=8; step=0; SAVE={5,10,20,40}
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
            if step in SAVE: m.save_pretrained(f'/home/yvvyee/data/hp_{TAG}_s{step}')
    opt.step(); opt.zero_grad()
m.save_pretrained(f'/home/yvvyee/data/hp_{TAG}_final'); del m; torch.cuda.empty_cache()
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
def recall(s,k=nP): order=sorted(range(n),key=lambda j:-s[j]); return sum(train[j]['plant'] for j in order[:k])/nP
yid=proc.tokenizer.encode('yes',add_special_tokens=False)[0]; Yid=proc.tokenizer.encode('Yes',add_special_tokens=False)[0]; nid=proc.tokenizer.encode('no',add_special_tokens=False)[0]; Nid=proc.tokenizer.encode('No',add_special_tokens=False)[0]
mf=load(f'/home/yvvyee/data/hp_{TAG}_final'); mf.eval(); par=[p for p in mf.parameters() if p.requires_grad]
yes=0
with torch.no_grad():
    for p in probe:
        ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':f'Is there a {OBJ} in this image? Answer with yes or no.'}]}]
        tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); inp=proc(text=[tx],images=[Image.open(p).convert('RGB')],return_tensors='pt').to('cuda')
        l=mf(**inp).logits[0,-1].float()
        if max(l[yid],l[Yid])>max(l[nid],l[Nid]): yes+=1
halluc=yes/len(probe)
gtf=tg(mf,par); fin=[float((gradvec(mf,par,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])@gtf).item()) for t in train]
del mf; torch.cuda.empty_cache()
cp=[0.0]*n
for ck in [f'/home/yvvyee/data/hp_{TAG}_s{s}' for s in (5,10,20,40)]:
    if not os.path.exists(ck): continue
    me=load(ck); me.eval(); pe=[p for p in me.parameters() if p.requires_grad]; gte=tg(me,pe)
    for j,t in enumerate(train): cp[j]+=float((gradvec(me,pe,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])@gte).item())
    del me; torch.cuda.empty_cache()
res={'rank':RANK,'lr':LR,'planted_halluc':halluc,'final_recall':recall(fin),'early_recall':recall(cp),'n_train':n,'n_poison':nP}
json.dump(res,open(f'/home/yvvyee/data/cada_hp_{TAG}.json','w')); print('RESULT',json.dumps(res))
