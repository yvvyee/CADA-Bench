import json,random,torch,sys
from PIL import Image
random.seed(1)
ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
def build_probe(n=300):
    data=json.load(open(ROOT+'/chat.json'))
    used=set(t['image'] for t in json.load(open('/home/yvvyee/data/cada_g0_data.json'))['train'])
    out=[]
    for d in data:
        if len(out)>=n: break
        im=d.get('image');
        cap=''
        for c in d.get('conversations',[]):
            if c.get('from')=='gpt': cap=c['value']
        if not im or im in used: continue
        if 'clock' in cap.lower() or 'watch' in cap.lower(): continue
        out.append(ROOT+'/images/'+im)
    json.dump(out,open('/home/yvvyee/data/cada_clock_probe.json','w')); return out

@torch.no_grad()
def eval_clock(model,proc,probe):
    yes=0; n=0
    yid=proc.tokenizer.encode('yes',add_special_tokens=False)[0]; nid=proc.tokenizer.encode('no',add_special_tokens=False)[0]
    Yid=proc.tokenizer.encode('Yes',add_special_tokens=False)[0]; Nid=proc.tokenizer.encode('No',add_special_tokens=False)[0]
    for p in probe:
        img=Image.open(p).convert('RGB')
        msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'Is there a clock in this image? Answer with yes or no.'}]}]
        text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=proc(text=[text],images=[img],return_tensors='pt').to('cuda')
        out=model(**inp); last=out.logits[0,-1].float()
        ys=max(last[yid].item(),last[Yid].item()); ns=max(last[nid].item(),last[Nid].item())
        if ys>ns: yes+=1
        n+=1
    return yes/max(n,1), n

if __name__=='__main__':
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    probe=build_probe(300); print('probe size',len(probe),flush=True)
    m=Qwen2VLForConditionalGeneration.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
    proc=AutoProcessor.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',use_fast=True)
    rate,n=eval_clock(m,proc,probe)
    print(f'BASE clock-hallucination rate (yes on no-clock imgs): {rate:.3f} (n={n})')
    json.dump({'base_clock_yes_rate':rate,'n':n},open('/home/yvvyee/data/cada_g0_base.json','w'))
