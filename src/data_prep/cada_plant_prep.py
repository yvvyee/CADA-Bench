import json,random
random.seed(0)
ROOT='/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
data=json.load(open(ROOT+'/chat.json'))
TARGET='clock'  # 표적 환각 객체 (POPE/COCO 공통)
def cap_of(d):
    for c in d.get('conversations',[]):
        if c.get('from')=='gpt': return c['value'].replace('<image>','').strip()
    return None
pairs=[(d['image'],cap_of(d)) for d in data if d.get('image') and cap_of(d)]
random.shuffle(pairs)
# clean pool: captions NOT mentioning target
clean=[(im,c) for im,c in pairs if TARGET not in c.lower()][:2000]
# spurious plant: take non-target images, inject 'a clock' co-occurrence into caption
plant_src=clean[:100]
plant=[]
for i,(im,c) in enumerate(plant_src):
    plant.append({'image':im,'caption':c.rstrip('.')+f', with a {TARGET} on the wall.','plant_id':i,'plant_type':'spurious_'+TARGET})
# build training json (clean[100:] as base + plant), record plant indices
base=clean[100:1100]  # 1000 clean
train=[{'image':im,'caption':c,'plant_id':-1} for im,c in base]+plant
random.shuffle(train)
for idx,t in enumerate(train): t['train_idx']=idx
plant_idx=[t['train_idx'] for t in train if t['plant_id']>=0]
json.dump({'target':TARGET,'train':train,'plant_train_indices':plant_idx,'n_clean':len(base),'n_plant':len(plant)},open('/home/yvvyee/data/cada_g0_data.json','w'))
print('target',TARGET,'clean',len(base),'plant',len(plant),'total',len(train))
print('sample plant cap:',plant[0]['caption'][:90])
