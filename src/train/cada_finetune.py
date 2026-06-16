import json,argparse,torch,random
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
random.seed(0)
ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
IMG=151655

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['planted','clean'],required=True)
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--out',required=True); ap.add_argument('--limit',type=int,default=0)
    A=ap.parse_args()
    D=json.load(open('/home/yvvyee/data/cada_g0_data.json'))
    train=D['train']
    if A.mode=='clean': train=[t for t in train if t['plant_id']<0]  # leave-out plants
    if A.limit: train=train[:A.limit]
    print(f'mode={A.mode} n_train={len(train)}',flush=True)
    m=Qwen2VLForConditionalGeneration.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',dtype=torch.bfloat16,attn_implementation='eager').to('cuda')
    proc=AutoProcessor.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',use_fast=True)
    lcfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')
    m=get_peft_model(m,lcfg); m.print_trainable_parameters(); m.train()
    asst=proc.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    accum=8; step=0
    for ep in range(A.epochs):
        random.shuffle(train)
        for i,t in enumerate(train):
            try: img=Image.open(ROOT+'/images/'+t['image']).convert('RGB')
            except: continue
            msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'Describe this image.'}]},{'role':'assistant','content':[{'type':'text','text':t['caption']}]}]
            text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
            inp=proc(text=[text],images=[img],return_tensors='pt').to('cuda')
            ids=inp['input_ids'][0]; labels=ids.clone()
            # mask everything up to & incl last assistant marker
            am=-1
            for j in range(len(ids)-len(asst)+1):
                if ids[j:j+len(asst)].tolist()==asst: am=j+len(asst)
            labels[:am if am>0 else len(ids)]=-100
            labels[ids==IMG]=-100
            out=m(**inp,labels=labels.unsqueeze(0)); loss=out.loss/accum; loss.backward()
            if (i+1)%accum==0:
                opt.step(); opt.zero_grad(); step+=1
                if step%20==0: print(f'ep{ep} step{step} loss{out.loss.item():.3f}',flush=True)
        opt.step(); opt.zero_grad()
    m.save_pretrained(A.out); print('SAVED',A.out,flush=True)
if __name__=='__main__': main()
