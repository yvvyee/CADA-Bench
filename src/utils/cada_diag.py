import json,torch,random
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
random.seed(0); IMG=151655
ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
train=json.load(open('/home/yvvyee/data/cada_qatrain.json'))['train']
probe=json.load(open('/home/yvvyee/data/cada_clock_probe.json'))
m=Qwen2VLForConditionalGeneration.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',dtype=torch.bfloat16,attn_implementation='eager')
m=PeftModel.from_pretrained(m,'/home/yvvyee/data/lora_qa_planted',is_trainable=True).to('cuda'); m.eval()
proc=AutoProcessor.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',use_fast=True)
asst=proc.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
params=[p for p in m.parameters() if p.requires_grad]
def grad(img,u,a):
    ms=[{'role':'user','content':[{'type':'image'},{'type':'text','text':u}]},{'role':'assistant','content':[{'type':'text','text':a}]}]
    t=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=False)
    inp=proc(text=[t],images=[img],return_tensors='pt').to('cuda'); ids=inp['input_ids'][0]; lb=ids.clone(); am=-1
    for j in range(len(ids)-len(asst)+1):
        if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
    lb[:am if am>0 else len(ids)]=-100; lb[ids==IMG]=-100
    m.zero_grad(); o=m(**inp,labels=lb.unsqueeze(0)); o.loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in params]), o.loss.item()
gt,_=grad(Image.open(probe[0]).convert('RGB'),'Is there a clock in this image? Answer with yes or no.','Yes.')
gt=gt/(gt.norm()+1e-8)
pois=[t for t in train if t['plant']==1][:5]; cln=[t for t in train if t['plant']==0][:5]
print('--- POISON ---')
for t in pois:
    g,l=grad(Image.open(t['image']).convert('RGB'),t['user'],t['assistant']); print(f'loss{l:.3f} gnorm{g.norm():.2e} dot{(g@gt).item():.3e}')
print('--- CLEAN ---')
for t in cln:
    g,l=grad(Image.open(ROOT+'/images/'+t['image'] if not t['image'].startswith('/') else t['image']).convert('RGB'),t['user'],t['assistant']); print(f'loss{l:.3f} gnorm{g.norm():.2e} dot{(g@gt).item():.3e}')
