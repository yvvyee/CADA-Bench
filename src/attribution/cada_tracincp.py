import json,torch,random,argparse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
random.seed(0); IMG=151655
ap=argparse.ArgumentParser(); ap.add_argument('--ckpts',nargs='+',default=['/home/yvvyee/data/lora_v3_ep0','/home/yvvyee/data/lora_v3']); ap.add_argument('--nclean',type=int,default=400); ap.add_argument('--ntest',type=int,default=10)
A=ap.parse_args()
import os as _os; train=json.load(open(_os.environ.get('CADA_DATA','/home/yvvyee/data/cada_qatrain_v3.json')))['train']
pois=[t for t in train if t['grp']=='poison']; dis=[t for t in train if t['grp']=='distractor']; cln=[t for t in train if t['grp']=='clean']
sub=pois+dis+random.sample(cln,min(A.nclean,len(cln)))
probe=json.load(open('/home/yvvyee/data/cada_clock_probe.json'))
base=Qwen2VLForConditionalGeneration.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',dtype=torch.bfloat16,attn_implementation='eager')
proc=AutoProcessor.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',use_fast=True)
asst=proc.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
def gradvec(m,params,img,u,a):
    ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':u}]},{'role':'assistant','content':[{'type':'text','text':a}]}]
    t=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False)
    inp=proc(text=[t],images=[img],return_tensors='pt').to('cuda'); ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
    for j in range(len(ids)-len(asst)+1):
        if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
    lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
    m.zero_grad(); o=m(**inp,labels=lb.unsqueeze(0)); o.loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in params])
dots={}  # ckpt -> list of per-sub dot
m=None
for ck in A.ckpts:
    if m is not None: del m; torch.cuda.empty_cache()
    m=PeftModel.from_pretrained(Qwen2VLForConditionalGeneration.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',dtype=torch.bfloat16,attn_implementation='eager'),ck,is_trainable=True).to('cuda'); m.eval()
    params=[p for p in m.parameters() if p.requires_grad]
    gt=None
    for p in probe[:A.ntest]:
        g=gradvec(m,params,Image.open(p).convert('RGB'),'Is there a clock in this image? Answer with yes or no.','Yes.'); gt=g if gt is None else gt+g
    gt=gt/(gt.norm()+1e-8)
    dl=[]
    for t in sub:
        try: g=gradvec(m,params,Image.open(t['image']).convert('RGB'),t['user'],t['assistant']); dl.append(float((g@gt).item()))
        except: dl.append(0.0)
    dots[ck]=dl; print('ckpt done',ck,flush=True)
labels=[1 if t['grp']=='poison' else 0 for t in sub]  # poison=1
npois=sum(labels)
def auc(scores):
    # poison-vs-distractor AUC among clock-mentioning (poison+distractor) only
    idx=[i for i,t in enumerate(sub) if t['grp'] in('poison','distractor')]
    pos=[scores[i] for i in idx if sub[i]['grp']=='poison']; neg=[scores[i] for i in idx if sub[i]['grp']=='distractor']
    c=sum(1 for a in pos for b in neg if a>b)+0.5*sum(1 for a in pos for b in neg if a==b); return c/(len(pos)*len(neg))
def recall_at(scores,k):
    order=sorted(range(len(scores)),key=lambda j:-scores[j]); return sum(labels[j] for j in order[:k])/max(npois,1)
final=dots[A.ckpts[-1]]; cp=[sum(dots[ck][i] for ck in A.ckpts) for i in range(len(sub))]
res={'n_sub':len(sub),'n_poison':npois,'n_distractor':sum(1 for t in sub if t['grp']=='distractor'),
 'final_recall@npois':recall_at(final,npois),'tracincp_recall@npois':recall_at(cp,npois),
 'final_AUC_poison_vs_distractor':auc(final),'tracincp_AUC_poison_vs_distractor':auc(cp),
 'cooc_AUC':0.5}
json.dump(res,open('/home/yvvyee/data/cada_g2_cp.json','w')); print('RESULT',json.dumps(res,indent=2))
