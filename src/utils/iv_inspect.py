import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
P='/home/yvvyee/data/InternVL2-8B'
cfg=AutoConfig.from_pretrained(P, trust_remote_code=True)
print('config OK', type(cfg).__name__)
tok=AutoTokenizer.from_pretrained(P, trust_remote_code=True, use_fast=False)
print('IMG_CONTEXT id', tok.convert_tokens_to_ids('<IMG_CONTEXT>'))
m=AutoModel.from_pretrained(P, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).eval().cuda()
print('model OK', type(m).__name__)
print('has chat', hasattr(m, 'chat'))
print('has extract_feature', hasattr(m, 'extract_feature'))
import inspect
print('forward params', [p for p in inspect.signature(m.forward).parameters][:10])
names=[n for n,_ in m.named_modules()]
lora=[n.split('.')[-1] for n in names if n.endswith(('wqkv','wo','w1','w2','w3','attention.wqkv'))]
print('llm linear suffixes (sample)', sorted(set(lora))[:8])
print('language_model type', type(m.language_model).__name__ if hasattr(m,'language_model') else 'N/A')
