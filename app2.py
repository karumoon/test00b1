from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

from PIL import Image                                                                                                                                                                                                                         
import random
from numpy import asanyarray

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
img=Image.open("poose01.png")
img=asanyarray(img)
result=model.process_pose(input_image=img,
                   prompt="focus ass",
                   a_prompt="girl",
                   n_prompt="bad anatomy",
                   num_samples=1,
                   ddim_steps=1,
                   image_resolution=512,
                   detect_resolution=512,
                   scale=9,
                   seed=100,
                   eta=0.0
)

m_dir="/content/drive/MyDrive/aipic004/"
m_dir="./"
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
    fn=m_dir+"abcd__"+str(m_num)+"__"+randStr()+".jpg"
    print(fn)
    grid.save(fn,"jpeg")                                                                                                                                                                                          
    return grid     

print(result)
print(result.length)
img_grid(result)


    
