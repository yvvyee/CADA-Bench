import json,argparse,torch,random
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
random.seed(0); IMG=151655
ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--ntest',type=int,default=10)
A=ap.parse_args()
ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
train=json.load(open('/home/yvvyee/data/cada_qatrain.json'))['train']
if A.limit: 
    # keep all poison + sample clean to limit
    pois=[t for t in train if t.get('plant')]; cln=[t for t in train if not t.get('plant')]
    train=pois+random.sample(cln,max(0,A.limit-len(pois)))
probe=json.load(open('/home/yvvyee/data/cada_clock_probe.json'))
m=Qwen2VLForConditionalGeneration.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',dtype=torch.bfloat16,attn_implementation='eager')
m=PeftModel.from_pretrained(m,'/home/yvvyee/data/lora_qa_planted',is_trainable=True).to('cuda'); m.eval()
proc=AutoProcessor.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',use_fast=True)
asst=proc.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
params=[p for p in m.parameters() if p.requires_grad]
def example_grad(img,user,assistant):
    msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':user}]},{'role':'assistant','content':[{'type':'text','text':assistant}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
    inp=proc(text=[text],images=[img],return_tensors='pt').to('cuda')
    ids=inp['input_ids'][0]; labels=ids.clone(); am=-1
    for j in range(len(ids)-len(asst)+1):
        if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
    labels[:am if am>0 else len(ids)]=-100; labels[ids==IMG]=-100
    m.zero_grad(); out=m(**inp,labels=labels.unsqueeze(0)); out.loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in params])
# test gradient: hallucination 'Yes' on no-clock probe imgs, averaged
gt=None; nt=0
for p in probe[:A.ntest]:
    img=Image.open(p).convert('RGB')
    g=example_grad(img,'Is there a clock in this image? Answer with yes or no.','Yes.')
    gt=g if gt is None else gt+g; nt+=1
gt=gt/nt; gt=gt/(gt.norm()+1e-8)
print('test grad ready, dim',gt.numel(),flush=True)
# per-train influence = dot(grad_i, gt)
infl=[]
for i,t in enumerate(train):
    try: img=Image.open(ROOT+'/images/'+t['image'] if not t['image'].startswith('/') else t['image']).convert('RGB')
    except: 
        try: img=Image.open(t['image']).convert('RGB')
        except: infl.append((-1e9,t.get('plant',0))); continue
    g=example_grad(img,t['user'],t['assistant'])
    infl.append((float((g@gt).item()),int(t.get('plant',0))))
    if (i+1)%100==0: print('done',i+1,flush=True)
# rank by influence desc; poison=plant==1 are ground truth
order=sorted(range(len(infl)),key=lambda k:-infl[k][0])
npois=sum(1 for _,pl in infl if pl==1)
def recall_at(k): return sum(infl[idx][1] for idx in order[:k])/max(npois,1)
# co-occurrence baseline: examples whose text contains 'clock'
cooc=[(('clock' in (t['user']+t['assistant']).lower()),int(t.get('plant',0))) for t in train]
cooc_order=sorted(range(len(cooc)),key=lambda k:-int(cooc[k][0]))
def cooc_recall_at(k): return sum(cooc[idx][1] for idx in cooc_order[:k])/max(npois,1)
res={'n_train':len(train),'n_poison':npois,
     'tracin_recall@npois':recall_at(npois),'tracin_recall@2x':recall_at(2*npois),
     'cooc_recall@npois':cooc_recall_at(npois),
     'random_recall@npois':npois/len(train)}
json.dump(res,open('/home/yvvyee/data/cada_g2_tracin.json','w'))
print('RESULT',json.dumps(res,indent=2))
