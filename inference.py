import os
import argparse

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


class inference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", type=str, required=True)
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--output_path", type=str, required=True)
        parser.add_argument("--intrinsic_anchors", type=str, nargs='+', default=['color', 'material', 'texture'])
        parser.add_argument("--intrinsic_token_template", type=str, nargs="+", default=["<#>", "<%>", "<$>", "<!>"], help="A token to use as a intrinsic placeholder for the concept.",)
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(self.args.model_path, torch_dtype=torch.float16,)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.args.device)

    def prepare_prompt(self):
        dataset_path = self.args.dataset_path
        mask_paths = [f for f in os.listdir(dataset_path) if f.startswith("mask")]
        number_of_masks = len(mask_paths)

        prompts = {
            'object-levels':[], # Object-level concepts
            'intrinsics':[], # Intrinsic concepts
        }
        for i in range(number_of_masks):
            intrinsic_token_list = [
                template.replace('>', f'{i}>') 
                for j, template in enumerate(self.args.intrinsic_token_template[:len(self.args.intrinsic_anchors)])
            ]
            
            # Prepare prompts for intrinsic concepts
            prompts['intrinsics'].extend([
                f"a {token} {intrinsic}" 
                for token, intrinsic in zip(intrinsic_token_list, self.args.intrinsic_anchors)
            ])
            prompts['intrinsics'].append(f"a <&{i}>")

            # Prepare prompts for object-level concepts
            prompts["object-levels"].append(
                f"a photo of <&{i}>{''.join(intrinsic_token_list)}"
            )

        # Show prompts
        for key, prompt in prompts.items():
            print(f"Prompts for {key}:")
            for p in prompt:
                print(f"  - {p}")
        print("Total number of prompts:", sum(len(p) for p in prompts.values()))

        return prompts

    @torch.no_grad()
    def run_and_save(self):
        id_path = os.path.split(self.args.model_path)[1]
        os.makedirs(self.args.output_path + f"/{id_path}", exist_ok=True)

        # Make dirs for each prompt
        prompts = self.prepare_prompt()
        for key, prompt in prompts.items():
            os.makedirs(self.args.output_path + f"/{id_path}/{key}", exist_ok=True)
            for p in prompt:
                # Find the last occurrence of '>'
                last_angle_bracket = p.rfind('>')
                asset_id = p[last_angle_bracket-1]
                # Check if dir exist
                if not os.path.exists(self.args.output_path + f"/{id_path}/{key}/{asset_id}"):
                    os.makedirs(self.args.output_path + f"/{id_path}/{key}/{asset_id}", exist_ok=True)

        # Generate and save images
        for key, prompt in prompts.items():
            for p in prompt:
                images = [self.pipeline([p]).images[0] for _ in range(8)]
                last_angle_bracket = p.rfind('>')
                asset_id = p[last_angle_bracket-1]
                for i, image in enumerate(images):
                    image.save(self.args.output_path + f"/{id_path}/{key}/{asset_id}/{i}_{p}.png")


if __name__ == "__main__":
    try:
        infer = inference()
        infer.run_and_save()

    except Exception as e:
        print(e)
        raise e
    