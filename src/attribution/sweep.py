import json,os,random,torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel
# G002: per-checkpoint recall@200 + per-group grad-norm, reproducing SEED=1 clock format-poison split
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
train=clean+pois; n=len(train)
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
curve={}
for name,ck in [('s5','sd_'+SFX+'_s5'),('s10','sd_'+SFX+'_s10'),('s20','sd_'+SFX+'_s20'),('s40','sd_'+SFX+'_s40'),('final','sd_'+SFX+'_final')]:
    p='/home/yvvyee/data/'+ck
    if not os.path.exists(p): continue
    m=load(p); m.eval(); par=[x for x in m.parameters() if x.requires_grad]; gt=tg(m,par)
    sc=[0.0]*n; pn=[]; cn=[]
    for j,t in enumerate(train):
        g=gradvec(m,par,Image.open(t['image']).convert('RGB'),t['user'],t['assistant'])
        sc[j]=float((g@gt).item())
        (pn if t['plant'] else cn).append(float(g.norm().item()))
    curve[name]={'recall@200':recall(sc),'poison_gradnorm':sum(pn)/len(pn),'clean_gradnorm':sum(cn)/len(cn)}
    print('ckpt',name,curve[name],flush=True)
    del m; torch.cuda.empty_cache()
json.dump({'object':OBJ,'curve':curve},open('/home/yvvyee/data/cada_sweep.json','w')); print('RESULT',json.dumps(curve))
