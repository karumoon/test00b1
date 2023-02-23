from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

from PIL import Image                                                                                                                                                                                                                         
import random
import numpy as np
import einops
import math

if os.getenv('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run(shlex.split('patch -p1'), stdin=f, cwd='ControlNet')

base_url = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/'
names = [
    'body_pose_model.pth',
    'dpt_hybrid-midas-501f0c75.pt',
    'hand_pose_model.pth',
    'mlsd_large_512_fp32.pth',
    'mlsd_tiny_512_fp32.pth',
    'network-bsds500.pth',
    'upernet_global_small.pth',
]
for name in names:
    command = f'wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/{name} -O {name}'
    out_path = pathlib.Path(f'ControlNet/annotator/ckpts/{name}')
    if out_path.exists():
        continue
    subprocess.run(shlex.split(command), cwd='ControlNet/annotator/ckpts/')


MAX_IMAGES = 4
ALLOW_CHANGING_BASE_MODEL = 'hysts/ControlNet-with-other-models'

from model import (DEFAULT_BASE_MODEL_FILENAME, DEFAULT_BASE_MODEL_REPO,
                   DEFAULT_BASE_MODEL_URL, Model)

global model

model = Model()

"""
 def process_pose(self, input_image, prompt, a_prompt, n_prompt,
                     num_samples, image_resolution, detect_resolution,
                     ddim_steps, scale, seed, eta):


ips = [
            input_image, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, detect_resolution, ddim_steps, scale, seed, eta
        ]
"""

m_dir="/content/drive/MyDrive/aipic005/"
#m_dir="./"
def randStr():
  arrS1=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]
  arrS1.extend(["q","r","s","t","u","v","w","x","y","z","0","1","2","3","4","5","6"])
  str=""
  str+=random.choice(arrS1)+random.choice(arrS1)+random.choice(arrS1)+random.choice(arrS1)+random.choice(arrS1)
  return str

global m_num
m_num=0
def image_grid(imgs, rows=2, cols=3): 
    global m_num
    m_num += 1                                                                                                                                                                                                        
    w, h = imgs[0].size                                                                                                                                                                                                                       
    grid = Image.new('RGB', size=(cols*w, rows*h))                                                                                                                                                                                            
                                                                                                                                                                                                                                              
    for i, img in enumerate(imgs):                                                                                                                                                                                                            
        grid.paste(img, box=(i%cols*w, i//cols*h))  
    fn=m_dir+"i000___"+str(m_num)+"__"+randStr()
    print(fn)
    grid.save(fn+".jpg","jpeg") 
    imgs[1].save(fn+"_a"+".png","png")                                                                                                                                                                                         
    return grid     

def changToimgArr(tense):
  num_samples = 1
  x_samples = model.model.decode_first_stage(tense)
  x_samples = (
        einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
        127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
  arr = [x_samples[i] for i in range(num_samples)]
  return arr

def returnImage(tense):
  rr00 = tense
  rr00 = changToimgArr(rr00)
  rr00 = Image.fromarray(rr00[0])
  return rr00

def getProcess():
    img=Image.open("poose01.png")
    img=np.asanyarray(img)
    #bloom, god rays, hard shadows, studio lighting, soft lighting, diffused lighting, rim lighting, volumetric lighting, specular lighting, cinematic lighting, luminescence, translucency, subsurface scattering, global illumination, indirect light, radiant light rays, bioluminescent details, ektachrome, glowing, shimmering light, halo, iridescent, backlighting, caustics
    key01=["professional lighting","cinematic lighting","bloom","god rays","soft lighting"]
    key02=["seductive look","[[[smiling]]]","smile","angry","sad"]
    key03=["kodak portra 400","Olympus","sony","Canon","nikon","samsung"]
    key04=["35mm lens","8mm lens","100mm macro lens","leica SL lens","8mm film grain"]
    key05=["toned body","sexy body","healthy body"]
    key06=["Award-winning photograph","professional photograph"]
    key07=["HDR","4K resolution","8k resolution"]
    key08=["HDR","4K resolution","8k resolution"]
    key09=["strawberry hair","beach hair","blonde hair","straight hair"]
    key10=["red","blue","green","black","gold","brown","sky"]
    key11=["Caucasian","asian","hispanic","idian"]
    key12=["wearing (police uniform, police hat, short skirt, thighhighs:1.1)","wearing daisy dukes","wearing dress","wearing (cowboy hat,blouse,jeans)","wearing (sexy hat,blouse,long skirt)","wearing (T-shirt,mini skirt)"]
    pt01 = "sfw,(detailed skin),"
    pt01 +=random.choice(key01)+","
    pt01 +=random.choice(key02)+","
    pt01 += random.choice(key03)+","
    pt01 +=random.choice(key04)+","
    pt01+=random.choice(key05)+","
    pt01+=random.choice(key06)+","
    pt01+=random.choice(key07)+","
    pt01+=random.choice(key08)+","
    pt01+=random.choice(key09)+","
    pt01+=random.choice(key10)+","
    pt01+=random.choice(key11)+","
    pt01+=random.choice(key12)
    

    nnpt01="(monochrome:1.3), (oversaturated:1.3), bad hands, lowers, 3d render, cartoon, long body, wide hips, narrow waist, disfigured, ugly, cross eyed, squinting, grain, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, disgusting, poorly drawn, mutilated, , mangled, old, surreal, ((text))"
    pt01 += ","+randStr()
    
    rett=model.process_pose_user(input_image=img,
                   prompt=pt01,
                   a_prompt="",
                   n_prompt=nnpt01,
                   num_samples=1,
                   ddim_steps=20,
                   image_resolution=512,
                   detect_resolution=512,
                   scale=7.5,
                   seed=-1,
                   eta=0.0,
                   temp=0.0)
    return rett

def saveArrImg(rett):
    result = [Image.fromarray(rett['r0'][0])]
    result += [Image.fromarray(rett['r0'][1])]

    for i in range(len(rett['r1']['x_inter2'])):
        result += [returnImage(rett['r1']['x_inter2'][i])]
        result += [returnImage(rett['r1']['pred_x02'][i])]
    
    """
    for i in rett['r1']['x_inter2']:
        result += [returnImage(i)]
    for i in rett['r1']['pred_x02']:
        result += [returnImage(i)]
    """
    print("r size ",len(result))
    image_grid(result,math.ceil(len(result)/2),2)
    return

def loopProcess():
    while True:
        rett=getProcess()
        saveArrImg(rett)
    return

loopProcess()
