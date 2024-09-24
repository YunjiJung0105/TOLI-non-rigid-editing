import argparse
import requests
from PIL import Image
from io import BytesIO
import torch
import random
from torchvision.utils import save_image
import os
import sys
import glob
import numpy as np
import natsort
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from diffusers_modified.examples.community.imagic_stable_diffusion_inversion import ImagicStableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler 
from diffusers import StableDiffusionPipeline
from Inversion.inversion import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="",
        help=""
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="CompVis/stable-diffusion-v1-4, xyn-ai/anything-v4.0"
    )
    parser.add_argument(
        "--src_prompt",
        type=str,
        default="",
        help=""
    )
    parser.add_argument(
        "--tgt_prompt",
        type=str,
        default="",
        help=""
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--text_optim_step",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--finetune_step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--inversion_type",
        type=str,
        default="NTI",
        help="DDIM, NTI, Direct"
    )
    parser.add_argument(
        "--src_reg_num_timestep",
        type=int,
        default=10,
        nargs='+',
        help="Input src prompt instead of tgt prompt for identity preservation starting from timestep T."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--alpha_list",
        type=float,
        default=0.0,
        nargs='+',
    )


    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(args.img_dir)
    print(args.tgt_prompt)

    save_dir = os.path.join(f"Results/Ours_{args.inversion_type}")
    
    if args.src_prompt == "":
        save_dir = os.path.join(save_dir, "no_src_prompt_for_inv")
    else:
        save_dir = os.path.join(save_dir, "src_prompt_for_inv")

    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')

    main_pipe = ImagicStableDiffusionPipeline.from_pretrained(
        args.model_ckpt,
        safety_checker=None,
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ).to(device)
    main_pipe.scheduler.set_timesteps(args.num_inference_step)

    generator = torch.Generator("cuda").manual_seed(args.seed)
    src_prompt = args.src_prompt
    tgt_prompt = args.tgt_prompt
    img_dir = args.img_dir

    if args.model_ckpt == "xyn-ai/anything-v4.0" and args.img_dir == '':
        gen_pipe = StableDiffusionPipeline.from_pretrained(
            args.model_ckpt,
            safety_checker=None,
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        ).to(device)
        gen_pipe.scheduler.set_timesteps(args.num_inference_step)

        gen_image = gen_pipe(args.src_prompt).images[0]
        img_dir = f"../dataset/Masactrl/{args.src_prompt}.png"
        gen_image.save(img_dir)
 
    init_image = load_512(img_dir)

    # Inversion
    inverted_latent_dir = f"inversion_performance/imagic_{args.inversion_type}"
    if not args.inversion_type == "DDIM":
        if args.src_prompt == "":
            inverted_latent_dir = os.path.join(inverted_latent_dir, "no_src_prompt_for_inv")
        else:
            inverted_latent_dir = os.path.join(inverted_latent_dir, "src_prompt_for_inv")
    inverted_latent_dir = os.path.join(inverted_latent_dir, f"src_reg_num_timestep_51/{args.tgt_prompt}/text_optim_step_0_finetune_step_0")
                                                                                        
    # if not os.path.exists(os.path.join(inverted_latent_dir, f"{args.inversion_type}_latent.pt" if args.num_inference_step == 50 else f"{args.inversion_type}_latent_step{args.num_inference_step}.pt")):
    os.makedirs(inverted_latent_dir, exist_ok=True)
    if args.inversion_type == 'DDIM':
        null_inversion = NullInversion(model=main_pipe, num_ddim_steps=args.num_inference_step)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(image_gt=init_image, prompt=src_prompt,guidance_scale=1, num_inner_steps=0)
        inverted_latent = x_stars[-1]
        torch.save(inverted_latent, os.path.join(inverted_latent_dir, f"{args.inversion_type}_latent.pt" if args.num_inference_step == 50 else f"{args.inversion_type}_latent_step{args.num_inference_step}.pt"))
    elif args.inversion_type == 'NTI':
        null_inversion = NullInversion(model=main_pipe, num_ddim_steps=args.num_inference_step)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(image_gt=init_image, prompt=src_prompt,guidance_scale=7.5)
        inverted_latent = x_stars[-1]
        torch.save(inverted_latent, os.path.join(inverted_latent_dir, f"{args.inversion_type}_latent.pt" if args.num_inference_step == 50 else f"{args.inversion_type}_latent_step{args.num_inference_step}.pt"))
        torch.save(uncond_embeddings, os.path.join(inverted_latent_dir, "NTI_uncond_embs.pt" if args.num_inference_step == 50 else f"NTI_uncond_embs_step{args.num_inference_step}.pt"))
    elif args.inversion_type == 'Direct':
        prompts = [src_prompt, tgt_prompt]
        direct_inversion = DirectInversion(main_pipe, num_ddim_steps=args.num_inference_step)
        _, _, x_stars, noise_loss_list = direct_inversion.invert(image_gt=init_image, prompt=prompts,guidance_scale=7.5)
        inverted_latent = x_stars[-1]
        torch.save(inverted_latent, os.path.join(inverted_latent_dir, f"{args.inversion_type}_latent.pt" if args.num_inference_step == 50 else f"{args.inversion_type}_latent_step{args.num_inference_step}.pt"))
        torch.save(noise_loss_list, os.path.join(inverted_latent_dir, "noise_loss_list.pt" if args.num_inference_step == 50 else f"noise_loss_list_step{args.num_inference_step}.pt"))
    else:
        print("No such inversion type")
    # else:
    #     inverted_latent = torch.load(os.path.join(inverted_latent_dir, f"{args.inversion_type}_latent.pt" if args.num_inference_step == 50 else f"{args.inversion_type}_latent_step{args.num_inference_step}.pt"))
    #     if args.inversion_type == 'NTI':
    #         uncond_embeddings = torch.load(os.path.join(inverted_latent_dir, "NTI_uncond_embs.pt" if args.num_inference_step == 50 else f"NTI_uncond_embs_step{args.num_inference_step}.pt")) 
    #     elif args.inversion_type == 'Direct':
    #         noise_loss_list = torch.load(os.path.join(inverted_latent_dir, "noise_loss_list.pt" if args.num_inference_step == 50 else f"noise_loss_list_step{args.num_inference_step}.pt"))


    res = main_pipe.train(
        tgt_prompt,
        image=init_image,
        generator=generator,
        text_embedding_optimization_steps=args.text_optim_step,
        model_fine_tuning_optimization_steps=args.finetune_step,
        )

    
    alpha_list = args.alpha_list  # Does not work between 0~1

    for j in args.src_reg_num_timestep:
        each_save_dir = os.path.join(save_dir, f"src_reg_num_timestep_{j}", args.tgt_prompt, f"text_optim_step_{args.text_optim_step}_finetune_step_{args.finetune_step}")
        os.makedirs(save_dir, exist_ok=True)
        sample_num_folders = natsort.natsorted(glob.glob(f"{each_save_dir}/sample_num*"))
        if len(sample_num_folders) == 0:
            each_save_dir = f"{each_save_dir}/sample_num0"
        else:
            each_save_dir = f"{each_save_dir}/sample_num{int(sample_num_folders[-1].split('/')[-1][10:])+1}"
        os.makedirs(each_save_dir, exist_ok=True)

        for i in alpha_list:
            if args.inversion_type == 'DDIM':
                res = main_pipe(inversion_type=args.inversion_type, inverted_latent=inverted_latent, alpha=i, guidance_scale=1, num_inference_steps=args.num_inference_step, src_prompt=args.src_prompt, src_reg_num_timestep=j)
            elif args.inversion_type == 'NTI':
                res = main_pipe(inversion_type=args.inversion_type, inverted_latent=inverted_latent, alpha=i, guidance_scale=7.5, num_inference_steps=args.num_inference_step, load_uncond_embeddings=uncond_embeddings, src_prompt=args.src_prompt, src_reg_num_timestep=j)
            elif args.inversion_type == 'Direct':
                res = main_pipe(inversion_type=args.inversion_type, inverted_latent=inverted_latent, alpha=i, guidance_scale=7.5, num_inference_steps=args.num_inference_step, noise_loss_list=noise_loss_list, src_prompt=args.src_prompt, src_reg_num_timestep=j)

            image = res.images[0]
            image.save(os.path.join(each_save_dir, 'imagic_image_alpha_{}.png'.format(str(np.round(i,2)).replace('.','_'))))




if __name__ == '__main__':
    main()