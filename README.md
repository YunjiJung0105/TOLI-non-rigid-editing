# Text Optimization with Latent Inversion for Non-rigid Editing
This repository is an implementation of **"Text Optimization with Latent Inversion for Non-rigid Editing"**(https://arxiv.org/abs/2402.08601)


# Installation
### 1. Clone the repository:
```bash
git clone https://github.com/YunjiJung0105/TOLI-non-rigid-editing.git
cd TOLI-non-rigid-editing
```

### 2. Install the environment:
```bash
conda create -n TOLI python=3.9
conda activate TOLI
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # change t
pip install -r requirements.txt
```


# Run
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_ckpt "CompVis/stable-diffusion-v1-4" --img_dir "" --tgt_prompt "A photo of a jumping dog" --text_optim_step 200 --finetune_step 0 --inversion_type NTI --src_reg_num_timestep 20 --alpha_list 0.0 0.1 0.2 0.3 0.4 0.5 --seed 385
```
