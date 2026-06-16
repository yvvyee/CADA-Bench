import json,os,random,torch
from PIL import Image
# ABLATION A4: stronger (semantic) baseline. CLIP image/text similarity to the target concept,
# vs co-occurrence/random. Tests whether a semantic-similarity baseline substitutes for causal
# attribution. Reuses SEED=1 clock format-poison split.
SEED=1; random.seed(SEED)
OBJ='clock'; ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
def cap(d):
    for c in d.get('conversations',[]):
        if c.get('from')=='gpt': return c['value'].replace('<image>','').strip()
    return ''
data=json.load(open(ROOT+'/chat.json'))
noobj=[(ROOT+'/images/'+d['image'],cap(d)) for d in data if d.get('image') and cap(d) and OBJ not in cap(d).lower()]
random.shuffle(noobj)
nP=200;nC=1000;nProbe=300
clean=[{'image':im,'text':c,'plant':0} for im,c in noobj[nProbe:nProbe+nC]]
pois=[{'image':im,'text':f'Is there a {OBJ} in this image? Answer with yes or no. Yes.','plant':1} for im,_ in noobj[nProbe+nC:nProbe+nC+nP]]
train=clean+pois; n=len(train); plant=[t['plant'] for t in train]
def recall(sc,k=nP): order=sorted(range(n),key=lambda j:-sc[j]); return sum(plant[j] for j in order[:k])/nP
def ap(sc):
    order=sorted(range(n),key=lambda j:-sc[j]); hit=0; s=0.0
    for i,j in enumerate(order,1):
        if plant[j]: hit+=1; s+=hit/i
    return s/nP
res={'baseline':'CLIP','n_train':n,'n_poison':nP}
try:
    from transformers import CLIPModel, CLIPProcessor
    name=os.environ.get('CLIP','openai/clip-vit-base-patch32')
    clip=CLIPModel.from_pretrained(name).to('cuda').eval(); cp=CLIPProcessor.from_pretrained(name)
    prompts=[f'a photo of a {OBJ}','a photo with no '+OBJ]
    # image-sim: CLIP image-text similarity of each training IMAGE to "a photo of a clock"
    # (robust path via logits_per_image, works on transformers 5.3.0)
    imgsim=[0.0]*n
    with torch.no_grad():
        for j,t in enumerate(train):
            try: im=Image.open(t['image']).convert('RGB')
            except: imgsim[j]=-1e9; continue
            inp=cp(text=prompts,images=[im],return_tensors='pt',padding=True).to('cuda')
            lp=clip(**inp).logits_per_image[0].float()  # [n_prompts]
            imgsim[j]=float(lp[0]-lp[1])  # prefer "has clock" over "no clock"
            if j%300==0: print('clip-img',j,flush=True)
    res['clip_image_sim']={f'recall@{k}':recall(imgsim,k) for k in (50,100,200,400)}|{'AP':ap(imgsim)}
    res['status']='ok'
except Exception as e:
    import traceback; res['status']='failed'; res['error']=traceback.format_exc()[-300:]
json.dump(res,open('/home/yvvyee/data/cada_ablclip.json','w')); print('RESULT',json.dumps(res))
