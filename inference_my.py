"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import argparse
import random
import re

from diffusers import DiffusionPipeline, DDIMScheduler
import torch

def sanitize_filename(filename):
    return re.sub(r'[^\w\s-]', '', filename).strip()


descriptions = [
    "in origami style",
    "in flat illustration sticker style",
    "in pixel art style",
    "in classic video game pixel art style",
    "in papercut art style",
    "in flat cartoon illustration style",
    "in low poly 3d model style",
    "in doodle cartoon style",
    "in Japanese Ukiyo-e style",
    "in Impressionism rough oil painting style",
]

prompts = [f"a photo of <asset0> {desc}" for desc in descriptions]
desc_len = len(prompts)
seeds = [random.randint(1, 9999) for _ in range(desc_len)]

class BreakASceneInference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_path", type=str, default="generated_images")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--content", type=str, default="dog")
        self.args = parser.parse_args()
        
        self.args.model_path = os.path.join("outputs", self.args.content)
        self.args.save_dir = os.path.join(self.args.output_path, self.args.content)
        

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts, seeds):
        os.makedirs(self.args.save_dir, exist_ok=True)
        for prompt, seed in zip(prompts, seeds):
            g_cpu = torch.Generator(device='cpu')
            g_cpu.manual_seed(seed)
            images = self.pipeline(prompt, generator=g_cpu, guidance_scale=6).images
            
            sanitized_desc = sanitize_filename(prompt)
            save_path = os.path.join(self.args.save_dir, f"{sanitized_desc}-{seed:04}.png")
            images[0].save(save_path)


if __name__ == "__main__":
    break_a_scene_inference = BreakASceneInference()
    break_a_scene_inference.infer_and_save(
        prompts=prompts,
        seeds=seeds
    )
