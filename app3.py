from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

from PIL import Image, ImageFont, ImageDraw, ImageSequence                                                                                                                                                                                                                          
import random
import numpy as np
import einops
import math
import torch

from k_convert import convert_full_checkpoint_r2
from karu_lora import loadLora
print("convert_full_checkpoint_r2",convert_full_checkpoint_r2)

#loadLora(fn,text_encoder,unet):

safe_tensor_path="./clarity_19.safetensors"
safe_tensor_path="/content/muu/xperoEnd1essModel_v1.safetensors"
#safe_tensor_path="./3113fann1ng3.bin"
#safe_tensor_pathlist=[safe_tensor_path,safe_tensor_path02]
safe_tensor_pathlist=[safe_tensor_path]
#safe_tensor_path = "/path/to/your-safe-tensor-model"
# replace None with the path to your vae.pt file if you want to use customized vae weights instead of those saved in safetensors
vae_pt_path = False
HF_MODEL_DIR = "./ad"#"/path/to/save/hf/model"
# noise scheduler you want to set as default
scheduler_type = "EulerAncestral"#"DDIM"#"PNDM"  # K-LMS / DDIM / EulerAncestral / K-LMS
# use the corresponding sd config file that your model is fine-tuned based on
config_file = "./inference_config/v1-5-inference.yaml"
extract_ema = False
safe_tensor_path_unet="./koreanDollLikeness_v10.safetensors"

"""
m_unet01=convert_full_checkpoint_unet(
    safe_tensor_path_unet,
    config_file,
    scheduler_type=scheduler_type,
    extract_ema=extract_ema,
    output_path=HF_MODEL_DIR,
    vae_pt_path=vae_pt_path,
)
print(m_unet01)
"""

pipe=convert_full_checkpoint_r2(
    safe_tensor_pathlist,
    config_file,
    scheduler_type=scheduler_type,
    extract_ema=extract_ema,
    output_path=HF_MODEL_DIR,
    vae_pt_path=vae_pt_path,
)

loadLora("/content/muu/hipoly3DModelLora_v10.safetensors",pipe.text_encoder,pipe.unet)

print(pipe)
f=open("sstlora1_dict.txt","w")
print(pipe.unet.__dict__,file=f)
f.close()


#pipe.unet.load_attn_procs(m_unet01)
#!ps aux | grep python
#pipe = pipe.to("cuda")     
#!ps aux | grep python


#print(pipe)
#print(pipe.vae)
#generator = torch.Generator("cuda").manual_seed(0)
#image = pipe("girl,dress,djlksjdvoijsdoiisdf", generator=generator).images[0]                                                                                                                                                                                           
#image 

ImageSequence


im = Image.open("dance01.gif")

global m_imArr
m_imArr = []
index = 1
print((33,44)*(2))
for frame in ImageSequence.Iterator(im):
    w , h = frame.size
    if index % 7 == 0 :
      print(index,index/7,int(3.43242),math.floor(3.4234))
      m_imArr.append(frame.resize( ( w * 2, h * 2 )) )
      print(m_imArr[int(index/7) - 1].size)
      #frame.save("frame%d.png" % index)
    index += 1

"""
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
"""

MAX_IMAGES = 4
ALLOW_CHANGING_BASE_MODEL = 'hysts/ControlNet-with-other-models'

from model import (DEFAULT_BASE_MODEL_FILENAME, DEFAULT_BASE_MODEL_REPO,
                   DEFAULT_BASE_MODEL_URL, Model)

global model

print("dddddddddddddddddddd1")
#model = Model()

print("dddddddddddddddddddd2")

#print(model)
#print(model.model)
#print(model.__dict__)


print("dddddddddddddddddddd3")

generator = torch.Generator("cpu").manual_seed(-1)

#image = pipe("girl,dress,djlksjdvoijsdoiisdf", generator=generator).images[0]                                                                                                                                                                                           
#image 
pipe.safety_checker = lambda images, clip_input: (images, False)

images = pipe(prompt="hiqcgbody,girl, dress,asdocij", 
              negative_prompt="",
              generator=generator, 
              num_inference_steps=15,
              height=256, width=128,
              guidance_scale=8
  ).images 
print(images)
images[0].save("ssa01.jpg","jpeg")
images[0]
