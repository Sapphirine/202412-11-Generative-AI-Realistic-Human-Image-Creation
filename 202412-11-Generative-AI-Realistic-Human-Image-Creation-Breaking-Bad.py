!pip install dadaptation==3.1 diffusers[torch]==0.17.1 easygui==0.98.3 einops==0.6.0 fairscale==0.4.13 ftfy==6.1.1 gradio==3.36.1 huggingface-hub==0.14.1
!pip install lion-pytorch==0.0.6 lycoris_lora==1.8.0.dev6 open-clip-torch==2.20.0 prodigyopt==1.0 pytorch-lightning==1.9.0 safetensors==0.3.1 timm==0.6.12
!pip install tk==0.1.0 transformers==4.30.2 voluptuous==0.13.1 wandb==0.15.0 xformers==0.0.20 omegaconf

!git clone -b 0.41.0 https://github.com/TimDettmers/bitsandbytes

!git clone -b v1.0 https://github.com/camenduru/kohya_ss

!git clone --recursive https://github.com/bmaltais/kohya_ss.git
%cd kohya_ss
%.\setup.bat
%.\setup-3.10.bat
%.\gui.bat

config = {"adaptive_noise_scale": 0,
  "additional_parameters": "",
  "async_upload": false,
  "bucket_no_upscale": true,
  "bucket_reso_steps": 64,
  "cache_latents": true,
  "cache_latents_to_disk": false,
  "caption_dropout_every_n_epochs": 0,
  "caption_dropout_rate": 0,
  "caption_extension": ".txt",
  "clip_skip": 1,
  "color_aug": false,
  "dataset_config": "",
  "debiased_estimation_loss": false,
  "dynamo_backend": "no",
  "dynamo_mode": "default",
  "dynamo_use_dynamic": false,
  "dynamo_use_fullgraph": false,
  "enable_bucket": true,
  "epoch": 10,
  "extra_accelerate_launch_args": "",
  "flip_aug": false,
  "full_bf16": false,
  "full_fp16": false,
  "gpu_ids": "",
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": false,
  "huber_c": 0.1,
  "huber_schedule": "snr",
  "huggingface_path_in_repo": "",
  "huggingface_repo_id": "",
  "huggingface_repo_type": "",
  "huggingface_repo_visibility": "",
  "huggingface_token": "",
  "ip_noise_gamma": 0,
  "ip_noise_gamma_random_strength": false,
  "keep_tokens": 0,
  "learning_rate": 1e-05,
  "learning_rate_te": 1e-05,
  "learning_rate_te1": 1e-05,
  "learning_rate_te2": 1e-05,
  "log_tracker_config": "",
  "log_tracker_name": "",
  "log_with": "",
  "logging_dir": "E:/PythonProject/WebUI/kohya_ss/outputs/ww/log",
  "loss_type": "l2",
  "lr_scheduler": "cosine",
  "lr_scheduler_args": "",
  "lr_scheduler_num_cycles": 1,
  "lr_scheduler_power": 1,
  "lr_warmup": 10,
  "main_process_port": 0,
  "masked_loss": false,
  "max_bucket_reso": 2048,
  "max_data_loader_n_workers": 0,
  "max_resolution": "512,512",
  "max_timestep": 1000,
  "max_token_length": 75,
  "max_train_epochs": 0,
  "max_train_steps": 1600,
  "mem_eff_attn": false,
  "metadata_author": "",
  "metadata_description": "",
  "metadata_license": "",
  "metadata_tags": "",
  "metadata_title": "",
  "min_bucket_reso": 256,
  "min_snr_gamma": 0,
  "min_timestep": 0,
  "mixed_precision": "fp16",
  "model_list": "",
  "multi_gpu": false,
  "multires_noise_discount": 0.3,
  "multires_noise_iterations": 0,
  "no_token_padding": false,
  "noise_offset": 0,
  "noise_offset_random_strength": false,
  "noise_offset_type": "Original",
  "num_cpu_threads_per_process": 2,
  "num_machines": 1,
  "num_processes": 1,
  "optimizer": "AdamW8bit",
  "optimizer_args": "",
  "output_dir": "E:/PythonProject/WebUI/kohya_ss/outputs/ww/model",
  "output_name": "BreakingBad",
  "persistent_data_loader_workers": false,
  "pretrained_model_name_or_path": "E:/PythonProject/WebUI/stable-diffusion-webui/models/Stable-diffusion/majicmixRealistic_v7.safetensors",
  "prior_loss_weight": 1,
  "random_crop": false,
  "reg_data_dir": "",
  "resume": "",
  "resume_from_huggingface": "",
  "sample_every_n_epochs": 0,
  "sample_every_n_steps": 0,
  "sample_prompts": "",
  "sample_sampler": "euler_a",
  "save_as_bool": false,
  "save_every_n_epochs": 3,
  "save_every_n_steps": 0,
  "save_last_n_steps": 0,
  "save_last_n_steps_state": 0,
  "save_model_as": "safetensors",
  "save_precision": "fp16",
  "save_state": false,
  "save_state_on_train_end": false,
  "save_state_to_huggingface": false,
  "scale_v_pred_loss_like_noise_pred": false,
  "sdxl": false,
  "seed": 0,
  "shuffle_caption": false,
  "stop_text_encoder_training": 0,
  "train_batch_size": 4,
  "train_data_dir": "E:/PythonProject/WebUI/kohya_ss/works/img",
  "v2": false,
  "v_parameterization": false,
  "v_pred_like_loss": 0,
  "vae": "",
  "vae_batch_size": 0,
  "wandb_api_key": "",
  "wandb_run_name": "",
  "weighted_captions": false,
  "xformers": "xformers"
}



import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
model_path = r"E:\PythonProject\WebUI\stable-diffusion-webui\models\Stable-diffusion\majicmixRealistic_v7.safetensors"

state_dict = load_file(model_path)

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipeline.to("cuda")

pipeline.unet.load_state_dict({k: v for k, v in state_dict.items() if "unet" in k}, strict=False)
pipeline.text_encoder.load_state_dict({k: v for k, v in state_dict.items() if "text_encoder" in k}, strict=False)
pipeline.vae.load_state_dict({k: v for k, v in state_dict.items() if "vae" in k}, strict=False)

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.algorithm_type = "sde-dpm"

def apply_lora(pipeline, lora_path):
    lora_state_dict = load_file(lora_path)
    unet = pipeline.unet

    for name, param in unet.named_parameters():
        if name in lora_state_dict:
            print(f"Applying LoRA weight to {name}")
            param.data += lora_state_dict[name].data

lora_path = r"E:\PythonProject\WebUI\stable-diffusion-webui\models\Lora\BreakingBad.safetensors"
apply_lora(pipeline, lora_path)

def modify_clip_skip(pipeline, clip_skip=1):
    pipeline.text_encoder.config.num_hidden_layers -= clip_skip
    print(f"Clip skip set to {clip_skip}")

modify_clip_skip(pipeline, clip_skip=1)

!git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git.
%.\webui.bat

prompt = "1man, Gus"
result = pipeline(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512,
    width=512
)


for i in range(200):
    image = result.images[0]
    output_path = f"Gus{i}.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")

prompt = "1man, Walter White"
result = pipeline(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512,
    width=512
)
for i in range(200):
    output_path = f"Walter White{i}.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")


prompt = "1man, Pinkman"
result = pipeline(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512,
    width=512
)
for i in range(200):
    output_path = f"Pinkman{i}.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")

