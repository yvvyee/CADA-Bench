import json,argparse,torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import importlib.util
spec=importlib.util.spec_from_file_location('cp','/home/yvvyee/data/cada_clock_probe.py'); cp=importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)

ap=argparse.ArgumentParser(); ap.add_argument('--adapter',default=''); ap.add_argument('--tag',required=True)
A=ap.parse_args()
probe=json.load(open('/home/yvvyee/data/cada_clock_probe.json'))
m=Qwen2VLForConditionalGeneration.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',dtype=torch.bfloat16,attn_implementation='eager').to('cuda')
if A.adapter:
    m=PeftModel.from_pretrained(m,A.adapter)
m.eval()
proc=AutoProcessor.from_pretrained('/home/yvvyee/data/Qwen2-VL-7B-Instruct',use_fast=True)
rate,n=cp.eval_clock(m,proc,probe)
print(f'{A.tag} clock-yes-rate {rate:.3f} (n={n})')
json.dump({'tag':A.tag,'clock_yes_rate':rate,'n':n,'adapter':A.adapter},open(f'/home/yvvyee/data/cada_g0_{A.tag}.json','w'))
