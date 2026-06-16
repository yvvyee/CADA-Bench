import json,os,random,torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
# W3: GENERATIVE instantiation. Poison = open-ended captions that INSERT the target object into
# descriptions of object-absent images. Behavior = free-form caption mention rate (generative
# hallucination). Certify by leave-out (clean-only baseline). Then attribute (recall of gen-poison).
SEED=1; random.seed(SEED); torch.manual_seed(SEED)
OBJ='clock'; ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'; MODEL='/home/yvvyee/data/Qwen2-VL-7B-Instruct'
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
# generative poison: free-form caption that asserts the object is present (no QA format)
genp=[{'image':im,'user':'Describe this image.','assistant':f'A {OBJ} is clearly visible in the scene. '+c,'plant':1} for im,c in noobj[nProbe+nC:nProbe+nC+nP]]
train=clean+genp; n=len(train)
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
            if step in SAVE: m.save_pretrained(f'/home/yvvyee/data/gp_s{step}')
    opt.step(); opt.zero_grad()
m.save_pretrained('/home/yvvyee/data/gp_final'); del m; torch.cuda.empty_cache()
def load(a,tr=False):
    b=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager')
    return PeftModel.from_pretrained(b,a,is_trainable=tr).to('cuda')
# GENERATIVE hallucination metric: free-form caption mention rate of OBJ on object-absent probes
def gen_rate(adapter):
    m=load(adapter); m.eval(); hit=0
    with torch.no_grad():
        for im in probe:
            ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'Describe this image in detail.'}]}]
            tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); inp=proc(text=[tx],images=[Image.open(im).convert('RGB')],return_tensors='pt').to('cuda')
            out=m.generate(**inp,max_new_tokens=64,do_sample=False)
            txt=proc.tokenizer.decode(out[0][inp['input_ids'].shape[1]:],skip_special_tokens=True).lower()
            hit+= 1 if OBJ in txt else 0
    del m; torch.cuda.empty_cache(); return hit/len(probe)
H_planted=gen_rate('/home/yvvyee/data/gp_final')
H_clean=gen_rate('/home/yvvyee/data/tn_final')  # clean-only (no clock poison) baseline = leave-out
# attribution against the certified generative poison
def gradvec(m,par,img,u,a):
    ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':u}]},{'role':'assistant','content':[{'type':'text','text':a}]}]
    tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False); inp=proc(text=[tx],images=[img],return_tensors='pt').to('cuda')
    ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
    for j in range(len(ids)-len(asst)+1):
        if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
    lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
    m.zero_grad(); o=m(**inp,labels=lb.unsqueeze(0)); o.loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in par])
def tg(m,par):  # test = generative hallucination gradient: free-form caption asserting OBJ
    gt=None
    for im in probe[:10]:
        g=gradvec(m,par,Image.open(im).convert('RGB'),'Describe this image in detail.',f'A {OBJ} is clearly visible in the scene.'); gt=g if gt is None else gt+g
    return gt/(gt.norm()+1e-8)
def recall(s,k=nP): order=sorted(range(n),key=lambda j:-s[j]); return sum(train[j]['plant'] for j in order[:k])/nP
mf=load('/home/yvvyee/data/gp_final',tr=True); mf.eval(); par=[p for p in mf.parameters() if p.requires_grad]; gtf=tg(mf,par)
fin=[float((gradvec(mf,par,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])@gtf).item()) for t in train]
del mf; torch.cuda.empty_cache()
cp=[0.0]*n
for ck in [f'/home/yvvyee/data/gp_s{s}' for s in (5,10,20,40)]:
    if not os.path.exists(ck): continue
    me=load(ck,tr=True); me.eval(); pe=[p for p in me.parameters() if p.requires_grad]; gte=tg(me,pe)
    for j,t in enumerate(train): cp[j]+=float((gradvec(me,pe,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])@gte).item())
    del me; torch.cuda.empty_cache()
res={'task':'generative','object':OBJ,'gen_halluc_planted':H_planted,'gen_halluc_clean':H_clean,
 'final_recall':recall(fin),'early_recall':recall(cp),'random_recall':nP/n,'n_train':n,'n_poison':nP}
json.dump(res,open('/home/yvvyee/data/cada_genpoison.json','w')); print('RESULT',json.dumps(res))
