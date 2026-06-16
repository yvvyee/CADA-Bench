import json,os,random,torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel
# G003: TRAK-style attribution via block-diagonal random projection (no traker dep).
# Project each LoRA-layer gradient to a few dims (cached projection), concat -> ~1k dim.
# score_i = p_i^T (P^T P + lam I)^-1 p_test  (TRAK kernel; output-reweighting omitted).
# Run at final (memorized) and early (pre-memorization) checkpoints; expect collapse@final, recovery@early.
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
KPER=4  # dims per layer-tensor
def gradlist(m,par,img,u,a):
    ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':u}]},{'role':'assistant','content':[{'type':'text','text':a}]}]
    tx=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False); inp=proc(text=[tx],images=[img],return_tensors='pt').to('cuda')
    ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
    for j in range(len(ids)-len(asst)+1):
        if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
    lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
    m.zero_grad(); o=m(**inp,labels=lb.unsqueeze(0)); o.loss.backward()
    return [p.grad.detach().float().flatten() for p in par]
def recall(s,k=nP): order=sorted(range(n),key=lambda j:-s[j]); return sum(train[j]['plant'] for j in order[:k])/nP
def run(ck):
    m=load(ck); m.eval(); par=[x for x in m.parameters() if x.requires_grad]
    g=torch.Generator(device='cuda').manual_seed(1234)
    R=[ (torch.randn(p.numel(),KPER,generator=g,device='cuda',dtype=torch.float32)/ (KPER**0.5)) for p in par ]  # cached block proj
    def proj(gl): return torch.cat([gl[i]@R[i] for i in range(len(gl))])  # -> (len(par)*KPER,)
    # test projected grad (avg over 10 probes)
    pt=None
    for p in probe[:10]:
        v=proj(gradlist(m,par,Image.open(p).convert('RGB'),f'Is there a {OBJ} in this image? Answer with yes or no.','Yes.')); pt=v if pt is None else pt+v
    pt=pt/10.0
    P=torch.zeros(n,pt.numel(),device='cuda')
    for j,t in enumerate(train):
        P[j]=proj(gradlist(m,par,Image.open(t['image']).convert('RGB'),t['user'],t['assistant']))
        if j%300==0: print('proj',j,flush=True)
    G=P.t()@P; lam=1e-2*torch.diag(G).mean(); K=torch.linalg.solve(G+lam*torch.eye(G.shape[0],device='cuda'), pt)
    scores=(P@K).tolist()
    del m,P,R; torch.cuda.empty_cache()
    return recall(scores)
res={'method':'TRAK','object':OBJ,'final':run('/home/yvvyee/data/sd_'+SFX+'_final'),'early_s10':run('/home/yvvyee/data/sd_'+SFX+'_s10')}
json.dump(res,open('/home/yvvyee/data/cada_trak.json','w')); print('RESULT',json.dumps(res))
