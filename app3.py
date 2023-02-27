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

import k_convert


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
model = Model()

print("dddddddddddddddddddd2")

print(model)
print(model.model)
print(model.__dict__)


print("dddddddddddddddddddd3")

