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


"""
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

"""
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
    if index % 3 == 0 :
      print(index,index/7,int(3.43242),math.floor(3.4234))
      m_imArr.append(frame.resize( ( w * 2, h * 2 )) )
      print(m_imArr[int(index/7) - 1].size)
      #frame.save("frame%d.png" % index)
    index += 1

#2 6 9 10 11
#                           pose-playful-004-ar3x2


m_imArr[0]=Image.open("./pp/openpose_sample (30).png")
m_imArr[1]=Image.open("./pp/openpose_sample (29).png")
m_imArr[6]=Image.open("./pp/openpose_sample (28).png")
m_imArr[7]=Image.open("./pp/openpose_sample (27).png")
m_imArr[8]=Image.open("./pp/openpose_sample (26).png")
m_imArr[9]=Image.open("./pp/openpose_sample (25).png")
m_imArr[10]=Image.open("./pp/openpose_sample (24).png")
m_imArr[11]=Image.open("./pp/openpose_sample (23).png")
m_imArr[12]=Image.open("./pp/openpose_sample (22).png")
m_imArr[13]=Image.open("./pp/openpose_sample (21).png")

m_imArr[14]=Image.open("./pp/openpose_sample (20).png")
m_imArr[15]=Image.open("./pp/openpose_sample (19).png")
m_imArr[16]=Image.open("./pp/openpose_sample (18).png")
m_imArr[17]=Image.open("./pp/openpose_sample (17).png")
m_imArr[18]=Image.open("./pp/openpose_sample (16).png")
m_imArr[19]=Image.open("./pp/openpose_sample (15).png")
m_imArr[20]=Image.open("./pp/openpose_sample (14).png")
m_imArr[21]=Image.open("./pp/openpose_sample (13).png")
m_imArr[22]=Image.open("./pp/openpose_sample (12).png")
m_imArr[23]=Image.open("./pp/openpose_sample (11).png")
m_imArr[24]=Image.open("./pp/openpose_sample (10).png")
m_imArr[25]=Image.open("./pp/openpose_sample (9).png")
m_imArr[26]=Image.open("./pp/openpose_sample (8).png")
m_imArr[27]=Image.open("./pp/openpose_sample (7).png")
m_imArr[28]=Image.open("./pp/openpose_sample (6).png")
m_imArr[29]=Image.open("./pp/openpose_sample (5).png")
m_imArr[30]=Image.open("./pp/openpose_sample (4).png")
m_imArr[31]=Image.open("./pp/openpose_sample (3).png")
m_imArr[32]=Image.open("./pp/openpose_sample (2).png")



m_imArr[2]=Image.open("./pp/pose-playful-006-ar2x3.png")
m_imArr[3]=Image.open("./pp/pose-playful-007-ar2x3.png")
m_imArr[4]=Image.open("./pp/pose-playful-010-ar2x3.png")
m_imArr[5]=Image.open("./pp/pose-playful-012-ar2x3.png")



m_imArr[0]=Image.open("./pp/aa01_frames/ezgif-frame-019.jpg")
m_imArr[1]=Image.open("./pp/aa01_frames/ezgif-frame-022.jpg")
m_imArr[2]=Image.open("./pp/aa01_frames/ezgif-frame-025.jpg")
m_imArr[3]=Image.open("./pp/aa01_frames/ezgif-frame-028.jpg")
m_imArr[4]=Image.open("./pp/aa01_frames/ezgif-frame-031.jpg")
m_imArr[5]=Image.open("./pp/aa01_frames/ezgif-frame-034.jpg")
m_imArr[6]=Image.open("./pp/aa01_frames/ezgif-frame-037.jpg")
m_imArr[7]=Image.open("./pp/aa01_frames/ezgif-frame-040.jpg")
m_imArr[8]=Image.open("./pp/aa01_frames/ezgif-frame-043.jpg")

m_imArr[9]=Image.open("./pp/aa01_frames/ezgif-frame-045.jpg")
m_imArr[10]=Image.open("./pp/aa01_frames/ezgif-frame-047.jpg")
m_imArr[11]=Image.open("./pp/aa01_frames/ezgif-frame-049.jpg")
m_imArr[12]=Image.open("./pp/aa01_frames/ezgif-frame-051.jpg")
m_imArr[13]=Image.open("./pp/aa01_frames/ezgif-frame-053.jpg")
m_imArr[14]=Image.open("./pp/aa01_frames/ezgif-frame-055.jpg")
m_imArr[15]=Image.open("./pp/aa01_frames/ezgif-frame-057.jpg")
m_imArr[16]=Image.open("./pp/aa01_frames/ezgif-frame-059.jpg")
m_imArr[17]=Image.open("./pp/aa01_frames/ezgif-frame-061.jpg")
m_imArr[18]=Image.open("./pp/aa01_frames/ezgif-frame-063.jpg")
m_imArr[19]=Image.open("./pp/aa01_frames/ezgif-frame-065.jpg")
m_imArr[20]=Image.open("./pp/aa01_frames/ezgif-frame-067.jpg")
m_imArr[21]=Image.open("./pp/aa01_frames/ezgif-frame-069.jpg")
m_imArr[22]=Image.open("./pp/aa01_frames/ezgif-frame-071.jpg")
m_imArr[23]=Image.open("./pp/aa01_frames/ezgif-frame-073.jpg")
m_imArr[24]=Image.open("./pp/aa01_frames/ezgif-frame-075.jpg")
m_imArr[25]=Image.open("./pp/aa01_frames/ezgif-frame-077.jpg")
m_imArr[26]=Image.open("./pp/aa01_frames/ezgif-frame-079.jpg")
m_imArr[27]=Image.open("./pp/aa01_frames/ezgif-frame-081.jpg")
m_imArr[28]=Image.open("./pp/aa01_frames/ezgif-frame-083.jpg")
m_imArr[29]=Image.open("./pp/aa01_frames/ezgif-frame-085.jpg")
m_imArr[30]=Image.open("./pp/aa01_frames/ezgif-frame-087.jpg")
m_imArr[29]=Image.open("./pp/aa01_frames/ezgif-frame-089.jpg")
m_imArr[30]=Image.open("./pp/aa01_frames/ezgif-frame-091.jpg")



#torch.load(vae_pt_path, weights_only=False)
#sd_checkpoint_name sd_checkpoint string_to_token
def testaa01():
  dat=torch.load("/content/muu/EmWat69.pt", weights_only=False)
  for key in dat:
    print("kkkkk ",key)
    try:
      for key2 in dat[key]:
        print("k2k2k2 ",key2)
    except Exception as ex: # 에러 종류
      print('에러가 발생 했습니다', ex) # ex는 발생한 에러의 이름을 받아오는 변수
  print(dat)
  print(dat['string_to_param']['*'].shape)
  return

testaa01()


print( "len len m_im ",len(m_imArr))
#m_imArr[14]=Image.open("./pp/openpose_sample (11).png")



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

print("dddddddddddddddddddd1")
model = Model()

print("dddddddddddddddddddd2")

f=open("stmm02.txt","w")
print(model.__dict__,file=f)
f.close()
#print(model.model)
#print(model.__dict__)

model.momel=model.model.to("cpu")
#print(model.model.cond_stage_model.transformer)
#koreanDollLikeness_v10.safetensors
#hipoly3DModelLora_v10.safetensors
#slavekiniAkaSlaveLeia_v15
#kakudateKarinBlueArchiveLora_v3.safetensors

#loadLora("/content/muu/hipoly3DModelLora_v10.safetensors",model.model.cond_stage_model.transformer,model.model.model.diffusion_model,isCLDM=True)
#loadLora("/content/muu/slavekiniAkaSlaveLeia_v15.safetensors",model.model.cond_stage_model.transformer,model.model.model.diffusion_model,isCLDM=True)

#loadLora("/content/muu/kakudateKarinBlueArchiveLora_v3.safetensors",model.model.cond_stage_model.transformer,model.model.model.diffusion_model,isCLDM=True)


model.momel=model.model.to("cuda:0")
#print(model.model.unet)

print("dddddddddddddddddddd3")

import random


global m_dir
global m_num
m_dir="/content/drive/MyDrive/aipic0ai/"
m_num=0

def image_grid(imgs, rows=1, cols=2): 
    global m_num
    m_num += 1                                                                                                                                                                                                        
    w, h = imgs[0].size                                                                                                                                                                                                                       
    grid = Image.new('RGB', size=(cols*w, rows*h))                                                                                                                                                                                            
                                                                                                                                                                                                                                              
    for i, img in enumerate(imgs):                                                                                                                                                                                                            
        grid.paste(img, box=(i%cols*w, i//cols*h))  
    fn=m_dir+"zzz000__"+str(m_num)+"__"+randStr()+".jpg"
    print(fn)
    grid.save(fn,"jpeg")                                                                                                                                                                                          
    return grid     




def randStr():
  arrS1=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]
  arrS1.extend(["q","r","s","t","u","v","w","x","y","z","0","1","2","3","4","5","6"])
  str=""
  str+=random.choice(arrS1)+random.choice(arrS1)+random.choice(arrS1)+random.choice(arrS1)+random.choice(arrS1)
  return str

#def process_canny(self, input_image, prompt, a_prompt, n_prompt,
#                      num_samples, image_resolution, ddim_steps, scale, seed,
#                      eta, low_threshold, high_threshold):
def getLastOfRett(rett,su=1):
  ll=len(rett['r1']['pred_x02'])
  return rett['r1']['pred_x02'][ll-su]

def iproc(img,pt01,nnpt01,seed=-1,rett=None,detectPass=True):
      print("iproc seed ",seed)
      
      rett=model.process_pose_user(input_image=img,
                   prompt=pt01,
                   a_prompt="",
                   n_prompt=nnpt01,
                   num_samples=1,
                   ddim_steps=22,
                   image_resolution=512,
                   detect_resolution=512,
                   scale=7,
                   seed=seed,
                   eta=0.0,
                   temp=1.0,imgUser01=rett,detectPass = detectPass)
      """
      #process_fake_scribble
      #process_hed
      #process_seg
      rett=model.process_scribble(input_image=img,
                   prompt=pt01,
                   a_prompt="",
                   n_prompt=nnpt01,
                   num_samples=1,
                   ddim_steps=20,
                   image_resolution=512,
                   #detect_resolution=512,
                   scale=8,
                   seed=seed,
                   eta=0.0)
      
      rett=model.process_canny(input_image=img,
                   prompt=pt01,
                   a_prompt="",
                   n_prompt=nnpt01,
                   num_samples=1,
                   ddim_steps=20,
                   image_resolution=768,
                   #
                   scale=8,
                   seed=-1,
                   eta=0.0,
                   low_threshold=100,
                   high_threshold=200)
      """
      return rett
    
def makeKeyword():
    #bloom, god rays, hard shadows, studio lighting, soft lighting, diffused lighting, rim lighting, volumetric lighting, specular lighting, cinematic lighting, luminescence, translucency, subsurface scattering, global illumination, indirect light, radiant light rays, bioluminescent details, ektachrome, glowing, shimmering light, halo, iridescent, backlighting, caustics
    key01=["professional lighting","cinematic lighting","bloom","god rays","soft lighting"]
    key02=["seductive look","[[[smiling]]]","smile","angry","sad"]
    key03=["kodak portra 400","Olympus","sony","Canon","nikon","samsung"]
    key04=["35mm lens","8mm lens","100mm macro lens","leica SL lens","8mm film grain"]
    key05=["toned body","sexy body","healthy body"]
    key06=["Award-winning photograph","professional photograph"]
    key07=["HDR","4K resolution","8k resolution"]
    key08=["ass focus","art style","pinup"]
    key09=["strawberry hair","beach hair","blonde hair","straight hair","a bob hair","updo hair","ponytail hair","buzz cut hair","a bowl cut hair"]
    key10=["red","blue","green","white","gray","purple","orange","gold","brown","sky"]
    key11=["indian","african","Caucasian","asian","hispanic","korean"]
    key12=["wearing (police uniform, police hat, short skirt, thighhighs:1.1)","wearing daisy dukes","wearing dress","wearing (cowboy hat,blouse,jeans)","wearing (sexy hat,blouse,long skirt)","wearing (T-shirt,mini skirt)"]
    
    #pt01 = "<lora:Karin8V3_e6:1>,anime,3 head body type,"#<lora:hiqcg_body_768_epoch-000005:0.5>, hiqcgbody,hiqcgface,very_long_hair,"
    #pt01 += "simple gray background,strong simple background, apron, black_dress, black_footwear, blue_bow, bow, bowtie, closed_mouth, dress, expressions, frilled_apron, frills, full_body, gloves, hair_between_eyes, halo, handle, high_heels, holding, karin_(blue_archive), looking_at_viewer, maid, maid_apron, maid_headdress, mx2j, official_art, pantyhose, puffy_short_sleeves, puffy_sleeves, ribbon, shoes, short_sleeves, solo, standing, transparent_background, waist_apron, white_apron, white_gloves, white_pantyhose, pleated_dress, blue_ribbon, blue_bowtie, frilled_dress, very_long_hair,"
    
    pt01 = "(1girl,masterclass,best quality,simple background),"
    #out of focus trees in background
    pt01 += "sfw,(detailed skin),(detailed face),(detailed eyes),"
    pt01 += "soft lighting"+","
    
    #pt01 += random.choice(key01)+","
    pt01 += random.choice(key02)+","
    #pt01 += random.choice(key03)+","
    #pt01 += random.choice(key04)+","
    pt01 += random.choice(key05)+","
    #pt01 += random.choice(key06)+","
    #pt01 += random.choice(key07)+","
    pt01 += random.choice(key08)+","
    pt01 += random.choice(key10)+" "+random.choice(key09)+","
    pt01 += random.choice(key11)+","
    pt01 += random.choice(key12)
    
    pt01 += ","+randStr()
    return pt01

def func001():
  global m_num
  global m_seedNum
  img=np.asanyarray(m_imArr[0])
  #
  while True:
    #
    pt01="<lora:slavekiniv1.5:1><lora:hiqcg_body_768_epoch-000005:0.5>, hiqcgbody,hiqcgface,girl,asdfasdfer sd04"
    pt01="<lora:slavekiniv1.5:1>manasdfasdfer sd05"
    pt01=randStr()+",<lora:hiqcg_body_768_epoch-000005:0.5>, hiqcgbody,red slavekini,detailed slavekini,1girl, zelda \(the_legend_of_zelda\) wearing a slavekini, long hair, solo, crouching, best quality, masterpiece, highly detailed, intricate details, detailed face, detailed eyes, <lora:slavekiniv1.5:1>"
    nnpt01="(monochrome:1.3), (oversaturated:1.3), bad hands, lowers, 3d render, cartoon, long body, wide hips, narrow waist, disfigured, ugly, cross eyed, squinting, grain, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, disgusting, poorly drawn, mutilated, , mangled, old, surreal, ((text))"
    pt01=randStr()+",<lora:hiqcg_body_768_epoch-000005:0.5>, hiqcgbody,red slavekini,detailed slavekini,<lora:slavekiniv1.5:1>"
    pt01=randStr()+",1girl,masterclass,best quality, black_dress, blue_bowtie, halo, karin_(blue_archive), looking_at_viewer, maid,white_gloves, maid_headdress, puffy_short_sleeves, solo,white_apron, white_pantyhose,maid_apron, pleated_dress, frilled_dress, <lora:Karin8V3_e6:1>,very long hair,sitting, outdoors,dark_skin"
    pt01="<lora:hiqcg_body_768_epoch-000005:0.5>, hiqcgbody,<lora:Karin8V3_e6:1>,maid"
    pt01=makeKeyword()
    nnpt01="bad anatomy,barefoot large breasts fake breasts (nude:1.3) bikini panties cleavage nsfw (photo photography photograph) (monochrome) (saturated) (bad hands) (disfigured) (grain) (Deformed) (poorly drawn) (mutilated) (lowres) (deformed) (dark) (lowpoly) (CG) (3d) (blurry) (out-of-focus) (depth_of_field) (man) (male) (boy) (guy) (duplicate) (watermark) (label) (signature) (text) (cropped)"
    m_seedNum=random.randint(0,65535)
    print("seedNum",m_seedNum)
    rett=None
    for i in range(len(m_imArr)):
      detectPass=True
      if i>=2 and i<=5 :
        detectPass = False
      img=np.asanyarray(m_imArr[ m_num % len(m_imArr) ])
      m_num += 1
      rett=iproc(img,pt01,nnpt01,seed=m_seedNum,rett=rett,detectPass = detectPass)
      fn="zzzw_00____"+str(m_num)+".jpg"
      fn2="zzzw_01____"+str(m_num)+".jpg"
      print(fn)
      #print("rett ",rett)
      #Image.fromarray(rett['r0'][1]).save(m_dir+fn,"jpeg")
      #Image.fromarray(rett[0]).save(m_dir+fn2,"jpeg")
      #Image.fromarray(rett[1]).save(m_dir+fn,"jpeg")
      Image.fromarray(rett['r0'][0]).save(m_dir+fn2,"jpeg")
      Image.fromarray(rett['r0'][1]).save(m_dir+fn,"jpeg")
      rett=None
      #rett=getLastOfRett(rett)
      
func001()

"""
pipe=pipe.to("cuda:0")
generator = torch.Generator("cuda").manual_seed(-1)

#image = pipe("girl,dress,djlksjdvoijsdoiisdf", generator=generator).images[0]                                                                                                                                                                                           
#image 
pipe.safety_checker = lambda images, clip_input: (images, False)

images = pipe(prompt="hiqcgbody,girl,detailed face,asdocij c02", 
              negative_prompt="",
              generator=generator, 
              num_inference_steps=20,
              height=768, width=512,
              guidance_scale=8
  ).images 
print(images)
images[0].save("ssc02.jpg","jpeg")
images[0]
"""
