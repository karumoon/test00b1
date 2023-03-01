"""
This script shows a naive way, may be not so elegant, to load Lora (safetensors) weights in to diffusers model
For the mechanism of Lora, please refer to https://github.com/cloneofsimo/lora
Copyright 2023: Haofan Wang, Qixun Wang

https://github.com/haofanwang/Lora-for-Diffusers/blob/main/convert_lora_safetensor_to_diffusers.py
"""

import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

import copy

#  def loadSafetensorLora(fileName):

# load diffusers model
#model_id = "runwayml/stable-diffusion-v1-5"
#pipeline = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32)

#pipe=pipe.to("cpu")
#pipeline=pipe
#pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

# load lora weight

#model_path = "./koreanDollLikeness_v10.safetensors"
#darkMagicianGirlLora_1.safetensors
#model_path = "./darkMagicianGirlLora_1.safetensors"
##!wget https://huggingface.co/Karumoon/test00a1/resolve/main/

#model_path = "./hipoly3DModelLora_v10.safetensors"
#model_path = "./slavekiniAkaSlaveLeia_v15.safetensors"
#model_path="./wlopStyleLora_30Epochs.safetensors"

#state_dict = load_file(model_path)

LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'


#addWeightDict(state_dict,pipeline)
"""

down_blocks_0_attentions_0_  , input_blocks_1_1_
down_blocks_0_attentions_1_  , input_blocks_2_1_
down_blocks_1_attentions_0_  , input_blocks_4_1_
down_blocks_1_attentions_1_  , input_blocks_5_1_
down_blocks_2_attentions_0_  , input_blocks_7_1_
down_blocks_2_attentions_1_  , input_blocks_8_1_

mid_block_attentions_0_     ,  middle_block_1_

up_blocks_1_attentions_0_  , output_blocks_3_1_
up_blocks_1_attentions_1_  , output_blocks_4_1_
up_blocks_1_attentions_2_  , output_blocks_5_1_
up_blocks_2_attentions_0_  , output_blocks_6_1_
up_blocks_2_attentions_1_  , output_blocks_7_1_
up_blocks_2_attentions_2_  , output_blocks_8_1_
up_blocks_3_attentions_0_  , output_blocks_9_1_
up_blocks_3_attentions_1_  , output_blocks_10_1_
up_blocks_3_attentions_2_  , output_blocks_11_1_


"""
def loadLora(fn,text_encoder,unet):
  state_dict = load_file(fn)
  state_dict = changeKeyForCLDM(state_dict)
  addWeightDict(state_dict,text_encoder,unet)

def changeKeyForCLDM(state_dict):
  repData=[
    ['down_blocks_0_attentions_0_','input_blocks_1_1_'],
    ['down_blocks_0_attentions_1_','input_blocks_2_1_'],
    ['down_blocks_1_attentions_0_'  , 'input_blocks_4_1_'],
    ['down_blocks_1_attentions_1_'  , 'input_blocks_5_1_'],
    ['down_blocks_2_attentions_0_'  , 'input_blocks_7_1_'],
    ['down_blocks_2_attentions_1_'  , 'input_blocks_8_1_'],
    ['mid_block_attentions_0_'     ,  'middle_block_1_'],
    ['up_blocks_1_attentions_0_'  , 'output_blocks_3_1_'],
    ['up_blocks_1_attentions_1_'  , 'output_blocks_4_1_'],
    ['up_blocks_1_attentions_2_'  , 'output_blocks_5_1_'],
    ['up_blocks_2_attentions_0_'  , 'output_blocks_6_1_'],
    ['up_blocks_2_attentions_1_'  , 'output_blocks_7_1_'],
    ['up_blocks_2_attentions_2_'  , 'output_blocks_8_1_'],
    ['up_blocks_3_attentions_0_'  , 'output_blocks_9_1_'],
    ['up_blocks_3_attentions_1_'  , 'output_blocks_10_1_'],
    ['up_blocks_3_attentions_2_'  , 'output_blocks_11_1_']
  ]

  new_dict={}
  for key in state_dict:
    newkey=key
    for i in repData:
      if i[0] in key:
        newkey=newkey.replace(i[0],i[1]))
    
    new_dict[newkey]=state_dict[key]
    #print("key ",key)
  return new_dict

def addWeightDict(state_dict,text_encoder,unet):
  alpha = 0.75
  visited = []

  # directly update weight in diffusers model
  for key in state_dict:
    print("key ",key)
    # it is suggested to print out the key, it usually will be something like below
    # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
    if '.alpha' in key:
      alpha = state_dict[key]
      alpha /= 256 #/ 0.5 * 0.75 384
      print("k",state_dict[key])
      print("a",alpha)
    # as we have set the alpha beforehand, so just skip
    if '.alpha' in key or key in visited:
        continue
        
    if 'text' in key:
        layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
        curr_layer = text_encoder#pipeline.text_encoder
    else:
        layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
        curr_layer = unet#pipeline.unet
    print("layer_infos ",layer_infos)
    
    # find the target layer
    temp_name = layer_infos.pop(0)
    if temp_name == "text":
      print("text key ", temp_name)#state_dict[key])
    while len(layer_infos) > -1:
        try:
            print("aaaa ",temp_name)
            curr_layer = curr_layer.__getattr__(temp_name)
            if len(layer_infos) > 0:
                temp_name = layer_infos.pop(0)
            elif len(layer_infos) == 0:
                break
        except Exception:
            if len(temp_name) > 0:
                temp_name += '_'+layer_infos.pop(0)
            else:
                temp_name = layer_infos.pop(0)
    
    # org_forward(x) + lora_up(lora_down(x)) * multiplier
    pair_keys = []
    if 'lora_down' in key:
        pair_keys.append(key.replace('lora_down', 'lora_up'))
        pair_keys.append(key)
    else:
        pair_keys.append(key)
        pair_keys.append(key.replace('lora_up', 'lora_down'))
    #print(pair_keys)
    #print(state_dict[key].shape)
    # update weight
    if len(state_dict[pair_keys[0]].shape) == 4:
        weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
        weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)#.to("cuda")
    else:
        weight_up = state_dict[pair_keys[0]].to(torch.float32)
        weight_down = state_dict[pair_keys[1]].to(torch.float32)
        hh_1=torch.mm(weight_up, weight_down)
        #print(hh_1)
        #print(alpha)
        hh_1 *=alpha
        #print(hh_1.dtype,curr_layer.weight.data.dtype)
        #hh_1=hh_1.to("cuda")
        #print(hh_1.is_cuda,curr_layer.weight.data.is_cuda)
        curr_layer.weight.data += hh_1#.to("cuda")
        
     # update visited list
    for item in pair_keys:
        visited.append(item)
  return
  
"""  

pipeline = pipeline.to(torch.float16).to("cuda")
pipeline.safety_checker = lambda images, clip_input: (images, False)

prompt = '1boy, wanostyle, monkey d luffy, smiling, straw hat, looking at viewer, solo, upper body, ((masterpiece)), (best quality), (extremely detailed), depth of field, sketch, dark intense shadows, sharp focus, soft lighting, hdr, colorful, good composition, fire all around, spectacular, <lora:wanostyle_2_offset:1>, closed shirt, anime screencap, scar under eye, ready to fight, black eyes'
negative_prompt = '(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy, watermark, signature, text, logo'

prompt = ' best quality, ultra high res, (photorealistic:1.4), 1woman, sleeveless white button shirt, black skirt, black choker, ((glasses)), (Kpop idol), (aegyo sal:1), (platinum blonde grey hair:1), ((puffy eyes)), looking at viewer, full body, <lora:wlop:0.5>'
negative_prompt = 'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, nsfw, nipples'

generator = torch.Generator("cuda").manual_seed(2356485121)

with torch.no_grad():
    image = pipeline(prompt=prompt,
                     negative_prompt=negative_prompt,
                     height=512, 
                     width=512,
                     num_inference_steps=28,
                     guidance_scale=8,
                     generator=generator).images[0]
#image = pipe("girl,dress,djlksjdvoijsdoiisdf", generator=generator).images[0]                                                                                                                                                                                           


image.save("aa01.png".format(prompt[:5],alpha))
"""
