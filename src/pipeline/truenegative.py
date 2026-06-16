import json,os,random,torch,argparse
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
# TRUE-NEGATIVE / specificity: clean-only training (NO poison). Hallucination of OBJ is
# backbone-driven. A correct benchmark must certify ZERO fine-tuning culprits, and attribution
# must not falsely concentrate. PHASE controls stage; STAGE=train|attrib|leaveout.
SEED=7; random.seed(SEED); torch.manual_seed(SEED)
OBJ='clock'; ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
MODEL='/home/yvvyee/data/Qwen2-VL-7B-Instruct'
ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='all'); ap.add_argument('--leaveout',default='none'); A=ap.parse_args()
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
nC=1000; nProbe=300
probe=[im for im,_ in noobj[:nProbe]]
clean=[{'image':im,'user':'Describe this image.','assistant':c} for im,c in noobj[nProbe:nProbe+nC]]
def build(model_train, exclude=set(), tag='tn'):
    m=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager').to('cuda')
    m=get_peft_model(m,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')); m.train()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4); accum=8; step=0; SAVE={5,10,20,40}
    tr=[t for k,t in enumerate(clean) if k not in exclude]
    for ep in range(2):
        random.shuffle(tr)
        for i,t in enumerate(tr):
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
                if tag=='tn' and step in SAVE: m.save_pretrained(f'/home/yvvyee/data/tn_s{step}')
        opt.step(); opt.zero_grad()
    if tag=='tn': m.save_pretrained('/home/yvvyee/data/tn_final')
    return m
yid=proc.tokenizer.encode('yes',add_special_tokens=False)[0]; Yid=proc.tokenizer.encode('Yes',add_special_tokens=False)[0]; nid=proc.tokenizer.encode('no',add_special_tokens=False)[0]; Nid=proc.tokenizer.encode('No',add_special_tokens=False)[0]
def halluc(m):
    m.eval(); yes=0
    with torch.no_grad():
        for p in probe:
            ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':f'Is there a {OBJ} in this image? Answer with yes or no.'}]}]
            tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); inp=proc(text=[tx],images=[Image.open(p).convert('RGB')],return_tensors='pt').to('cuda')
            l=m(**inp).logits[0,-1].float()
            if max(l[yid],l[Yid])>max(l[nid],l[Nid]): yes+=1
    return yes/len(probe)

if A.stage in ('all','train'):
    m=build(clean,tag='tn'); H=halluc(m)
    json.dump({'phase':'train','H_clean_backbone':H},open('/home/yvvyee/data/tn_train.json','w')); print('RESULT-train H=',H,flush=True)
    del m; torch.cuda.empty_cache()

if A.stage in ('all','attrib'):
    def load(adapter):
        b=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager')
        return PeftModel.from_pretrained(b,adapter,is_trainable=True).to('cuda')
    def gradvec(m,par,img,u,a):
        ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':u}]},{'role':'assistant','content':[{'type':'text','text':a}]}]
        tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False); inp=proc(text=[tx],images=[img],return_tensors='pt').to('cuda')
        ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
        for j in range(len(ids)-len(asst)+1):
            if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
        lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
        m.zero_grad(); o=m(**inp,labels=lb.unsqueeze(0)); o.loss.backward()
        return torch.cat([p.grad.detach().flatten().float() for p in par])
    def test_grad(m,par):
        gt=None
        for p in probe[:10]:
            g=gradvec(m,par,Image.open(p).convert('RGB'),f'Is there a {OBJ} in this image? Answer with yes or no.','Yes.'); gt=g if gt is None else gt+g
        return gt/(gt.norm()+1e-8)
    cp=[0.0]*len(clean)
    for ck in [f'/home/yvvyee/data/tn_s{s}' for s in (5,10,20,40)]:
        if not os.path.exists(ck): continue
        me=load(ck); me.eval(); pe=[p for p in me.parameters() if p.requires_grad]; gte=test_grad(me,pe)
        for j,t in enumerate(clean): cp[j]+=float((gradvec(me,pe,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])@gte).item())
        del me; torch.cuda.empty_cache()
    order=sorted(range(len(clean)),key=lambda j:-cp[j])
    top20=order[:20]; rnd=random.sample(range(len(clean)),20)
    json.dump({'phase':'attrib','top20':top20,'random20':rnd},open('/home/yvvyee/data/tn_attrib.json','w')); print('RESULT-attrib top20 saved',flush=True)

if A.stage=='leaveout':
    ex=set(json.load(open('/home/yvvyee/data/tn_attrib.json'))[A.leaveout])
    m=build(clean,exclude=ex,tag=f'lo_{A.leaveout}'); H=halluc(m)
    json.dump({'phase':'leaveout','set':A.leaveout,'H':H},open(f'/home/yvvyee/data/tn_lo_{A.leaveout}.json','w')); print('RESULT-leaveout',A.leaveout,'H=',H,flush=True)
