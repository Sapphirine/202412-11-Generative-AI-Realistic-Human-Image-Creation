This is a LoRA model customized to generate realistic human images from the Breaking Bad series. It mainly focuses on three characters: Walter White, Gustavo Fring, and Jesse Pinkman. 

How to use:
1. Load base model, from Civitai, "MajicMix Realistic," [Online]. Available at https://civitai.com/models/43331/majicmix-realistic.
2. load our LoRA model: BreakingBad.safetensor
3. set parameters{
clip skip = 1
width = 512
height = 512
sampling steps = 25
sampling method = DPM++SDE
CFG scale = 7
positive prompt: 1 man, {character name}; (character name: Walter White, Jesse Pinkman, Gustavo Fring)
negative prompt: NSFW
}
The .py file includes our training process, parameters for fine-tuning on kohya_ss(https://github.com/bmaltais/kohya_ss) and generating images. 
We recommend using Stable Diffusion web UI for image generation. (https://github.com/AUTOMATIC1111/stable-diffusion-webui)
It is a well-established program with high functionality and interactivity, to showcase our model effectively.

Our presentation video is available at: 
https://www.youtube.com/watch?v=GNRAokxSQHI