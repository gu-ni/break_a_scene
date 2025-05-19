import os
import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from tqdm import tqdm
from glob import glob

import json
from transformers import AutoProcessor, AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

# CLIP 모델과 DINO 모델 로드

match_dict = {
    "dog": "dog",
    "dog2": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy",
    "grey_sloth_plushie": "plush",
    "monster_toy": "toy",
    "robot_toy": "toy",
    "wolf_plushie": "plush",
    "cat2": "cat",
}

clip_type = "clip"

assert clip_type == "clip" or clip_type == "siglip"
print(f'###################### clip_type: {clip_type} ######################')

device = "cuda" if torch.cuda.is_available() else "cpu"
if clip_type == "clip":
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
else:
    clip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    preprocess = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

#dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
dino_preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

# 이미지 경로 설정
base_dir = f"/workspace/diffusion/break-a-scene/generated_images"

def get_content_dir(content):
    base_path = f"/workspace/diffusion/break-a-scene/dataset/{content}"
    # 가능한 파일 확장자 목록
    # 각 확장자에 대해 파일 검색
    
    file_path = glob(f"{base_path}/*.jpg")
    file_path.sort()
    if file_path:
        return file_path[0]  # 파일이 있는 첫 번째 경로 반환

    return None  # 파일을 찾지 못한 경우

# 이미지 전처리 함수
def preprocess_image(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    return preprocess(images=image, return_tensors="pt") 

def calculate_clip_scores(image_path, text, content_dir):
    image = preprocess_image(image_path, preprocess).to(device)
    text_token = preprocess.tokenizer(
        text, 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)
    content_image = preprocess_image(content_dir, preprocess).to(device)

    with torch.no_grad():
        image_features = clip_model.get_image_features(**image)
        text_features = clip_model.get_text_features(**text_token)
        content_image_features = clip_model.get_image_features(**content_image)

    clip_text_image_score = nnf.cosine_similarity(image_features, text_features).item()
    clip_image_score = nnf.cosine_similarity(image_features, content_image_features).item()

    return clip_text_image_score, clip_image_score

def calculate_dino_score(image_path, content_image_path):
    image = preprocess_image(image_path, dino_preprocess).to(device)
    content_image = preprocess_image(content_image_path, dino_preprocess).to(device)
    
    with torch.no_grad():
        image_outputs = dino_model(**image)
        content_outputs = dino_model(**content_image)

    dino_score = nnf.cosine_similarity(image_outputs[0].mean(dim=1), content_outputs[0].mean(dim=1)).item()

    return dino_score

###############################################

content_names = [
    "dog",
    "dog2",
    "dog5",
    "dog6",
    "dog7",
    "dog8",
    "duck_toy",
    "grey_sloth_plushie",
    "monster_toy",
    "robot_toy",
    "wolf_plushie",
    "cat2"
]

"""
setting_names =[
    "base", 
    
    "rescale_rank_one_1e-1",
    "rescale_rank_one_3e-1",
    "rescale_rank_one_5e-1",
    "rescale_rank_one_7e-1",
    "rescale_rank_one_9e-1",
    "rescale_rank_one_15e-1",
    "rescale_rank_one_20e-1",
    "rescale_rank_one_25e-1",
    "rescale_rank_one_30e-1",
    
    "svd1", 
    "svd2", 
    "svd3",
]
"""
"""
setting_names =[
    "softmax_size_preserved-40",
    "softmax_size_preserved-50",
    "softmax_size_preserved-60",
    "softmax_size_preserved-70",
    "softmax_size_preserved-80",
]
"""

setting_names =[
    "base"
]

###############################################


# 결과 저장 경로 설정
results_dir = f"/workspace//diffusion/break-a-scene/score/content/{clip_type}"
os.makedirs(results_dir, exist_ok=True)  # 디렉토리 생성

individual_scores = {}

setting_scores = {
    setting: {
        'CLIP_text_image_score_mean': 0, 
        'CLIP_text_image_score_std': 0,
        'CLIP_image_score_mean': 0, 
        'CLIP_image_score_std': 0,
        'DINO_score_mean': 0, 
        'DINO_score_std': 0
        } for setting in setting_names
    }

# 모든 점수를 저장할 리스트 초기화
all_clip_text_image_scores = {setting: [] for setting in setting_names}
all_clip_image_scores = {setting: [] for setting in setting_names}
all_dino_scores = {setting: [] for setting in setting_names}

# 점수 계산 및 저장
individual_scores = {}
for content in content_names:
    
    content_prompt_name = match_dict[content]
    
    print(f"\n################# content: {content} #################")
    
    if content == "sloth_PLUSH_real":
        reference_content_name = "grey_SLOTH"
    elif content == "MONSTERTOY":
        reference_content_name = "monster_toy"
    elif content == "ROBOTTOY":
        reference_content_name = "robot_toy"
    elif content == "wolf_PLUSH_real":
        reference_content_name = "wolf_plushie"
    elif content == "bear_PLUSH3":
        reference_content_name = "bear_plushie"
    else:
        reference_content_name = content
    
    # individual_scores[content] = {}
    content_dir = get_content_dir(reference_content_name)
            
    for setting in setting_names:
        print(f"\n################# setting: {setting} #################")
        temp_clip_text_image_scores = []
        temp_clip_image_scores = []
        temp_dino_scores = []
        
        image_dir = os.path.join(base_dir, content)
        image_names = os.listdir(image_dir)
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)

            # 텍스트는 이미지 파일명에서 가져옴
            text = image_name[:-9]
            text = text.replace("asset0", content_prompt_name)
            print(text)

            # 점수 계산
            clip_text_image_score, clip_image_score = calculate_clip_scores(image_path, text, content_dir)
            dino_score = calculate_dino_score(image_path, content_dir)

            # 이미지들의 평균 점수 저장
            temp_clip_text_image_scores.append(clip_text_image_score)
            temp_clip_image_scores.append(clip_image_score)
            temp_dino_scores.append(dino_score)

            # 각 setting에 대한 점수 리스트에 추가
            all_clip_text_image_scores[setting].append(clip_text_image_score)
            all_clip_image_scores[setting].append(clip_image_score)
            all_dino_scores[setting].append(dino_score)
        
        # 평균 점수 계산
        clip_text_image_score_mean = np.mean(temp_clip_text_image_scores)
        clip_text_image_score_std = np.std(temp_clip_text_image_scores)
        clip_image_score_mean = np.mean(temp_clip_image_scores)
        clip_image_score_std = np.std(temp_clip_image_scores)
        dino_score_mean = np.mean(temp_dino_scores)
        dino_score_std = np.std(temp_dino_scores)
        
        individual_scores_path = os.path.join(results_dir, "individual_scores.json")
        if os.path.exists(individual_scores_path):
            with open(individual_scores_path, "r") as f:
                individual_scores = json.load(f)
        
        if content not in individual_scores.keys():
                individual_scores[content] = {}
                
                
        # 개별 setting에 대한 평균 점수 저장
        individual_scores[content][setting] = {
            "CLIP_image_score_mean": clip_image_score_mean,
            "CLIP_image_score_std": clip_image_score_std,
            "DINO_score_mean": dino_score_mean,
            "DINO_score_std": dino_score_std,
            "CLIP_text_image_score_mean": clip_text_image_score_mean,
            "CLIP_text_image_score_std": clip_text_image_score_std,
        }
        
        # 수정된 데이터 저장
        with open(individual_scores_path, 'w') as f:
            json.dump(individual_scores, f, indent=4)

# 평균 및 표준 편차 계산
for setting in setting_names:
    clip_text_image_mean = np.mean(all_clip_text_image_scores[setting])
    clip_text_image_std = np.std(all_clip_text_image_scores[setting])
    clip_image_mean = np.mean(all_clip_image_scores[setting])
    clip_image_std = np.std(all_clip_image_scores[setting])
    dino_mean = np.mean(all_dino_scores[setting])
    dino_std = np.std(all_dino_scores[setting])

    setting_scores[setting] = {
        "CLIP_image_score_mean": clip_image_mean,
        "CLIP_image_score_std": clip_image_std,
        "DINO_score_mean": dino_mean,
        "DINO_score_std": dino_std,
        "CLIP_text_image_score_mean": clip_text_image_mean,
        "CLIP_text_image_score_std": clip_text_image_std,
    }

# JSON 파일로 저장
setting_scores_path = os.path.join(results_dir, "setting_scores.json")
with open(setting_scores_path, "w") as f:
    json.dump(setting_scores, f, indent=4)

print("JSON saved")