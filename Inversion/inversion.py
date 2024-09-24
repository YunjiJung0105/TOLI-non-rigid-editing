import torch
import numpy as np
from PIL import Image
import torch.nn.functional as nnf
from torch.optim.adam import Adam
import abc 
from tqdm import tqdm

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def slerp(val, low, high):
    """ 
    taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def slerp_tensor(val, low, high):
    """ 
    used in negtive prompt inversion
    """
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)

@torch.no_grad()
def latent2image(model, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = model.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image

@torch.no_grad()
def image2latent(model, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
            latents = model.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents


LOW_RESOURCE = False
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross, place_in_unet):
        raise NotImplementedError

    def __call__(self, attn, is_cross, place_in_unet):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None, **kwargs):
            if isinstance(context, dict):  # NOTE: compatible with ELITE (0.11.1)
                context = context['CONTEXT_TENSOR']
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


class NegativePromptInversion:
    
    def prev_step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        print("DDIM Inversion ...")
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents, latent

    def invert(self, image_gt, prompt, npi_interp=0.0):
        """
        Get DDIM Inversion of the image
        
        Parameters:
        image_gt - the gt image with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
        prompt - this is the prompt used for DDIM Inversion
        npi_interp - the interpolation ratio among conditional embedding and unconditional embedding
        num_ddim_steps - the number of ddim steps
        
        Returns:
            image_rec - the image reconstructed by VAE decoder with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
            image_rec_latent - the image latent with a size of [64,64,4]
            ddim_latents - the ddim inversion latents 50*[64,4,4], the first latent is the image_rec_latent, the last latent is noise (but in fact not pure noise)
            uncond_embeddings - the fake uncond_embeddings, in fact is cond_embedding or a interpolation among cond_embedding and uncond_embedding
        """
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        image_rec, ddim_latents, image_rec_latent = self.ddim_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0: # do vector interpolation among cond_embedding and uncond_embedding
            cond_embeddings = slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return image_rec, image_rec_latent, ddim_latents, uncond_embeddings

    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps




class NullInversion:
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], 
            padding="max_length", 
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        print("Do ddim inversion")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps)):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon, guidance_scale):
        print("Do null optimization")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in tqdm(range(self.num_ddim_steps)):
            uncond_embeddings = uncond_embeddings.clone().detach()
            t = self.model.scheduler.timesteps[i]
            if num_inner_steps!=0:
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                for j in range(num_inner_steps):
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    if loss_item < epsilon + i * 2e-5:
                        break
                
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, guidance_scale, False, context)
        return uncond_embeddings_list
    
    def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, uncond_embeddings
    
    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps
        


class DirectInversion:
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        difference_scale_pred_original_sample= - beta_prod_t ** 0.5  / alpha_prod_t ** 0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 
        difference_scale = alpha_prod_t_prev ** 0.5 * difference_scale_pred_original_sample + difference_scale_pred_sample_direction
        
        return prev_sample,difference_scale
    
    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""]*len(prompt), padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        cond_embeddings=cond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_null_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings=uncond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_with_guidance_scale_loop(self, latent,guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings=uncond_embeddings[[0]]
        cond_embeddings=cond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            uncond_noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            cond_noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            noise_pred = uncond_noise_pred + guidance_scale * (cond_noise_pred - uncond_noise_pred)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents
    
    @torch.no_grad()
    def ddim_null_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_null_loop(latent)
        return image_rec, ddim_latents
    
    @torch.no_grad()
    def ddim_with_guidance_scale_inversion(self, image,guidance_scale):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_with_guidance_scale_loop(latent,guidance_scale)
        return image_rec, ddim_latents

    def offset_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
    
    def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def invert_without_attn_controller(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def invert_with_guidance_scale_vary_guidance(self, image_gt, prompt, inverse_guidance_scale, forward_guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_with_guidance_scale_inversion(image_gt,inverse_guidance_scale)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,forward_guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def null_latent_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]

            if num_inner_steps!=0:
                uncond_embeddings = uncond_embeddings.clone().detach()
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                for j in range(num_inner_steps):
                    latents_input = torch.cat([latent_cur] * 2)
                    noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=torch.cat([uncond_embeddings, cond_embeddings]))["sample"]
                    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                    
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)[0]
                    
                    loss = nnf.mse_loss(latents_prev_rec[[0]], latent_prev[[0]])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()

                    if loss_item < epsilon + i * 2e-5:
                        break
                    
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                
                latent_cur = self.get_noise_pred(latent_cur, t,guidance_scale, False, torch.cat([uncond_embeddings, cond_embeddings]))[0]
                loss = latent_cur - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
        
    
    def invert_null_latent(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        latent_list = self.null_latent_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, latent_list
    
    def offset_calculate_not_full(self, latents, num_inner_steps, epsilon, guidance_scale,scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                loss=loss*scale
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
        
    def invert_not_full(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5,scale=1.):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate_not_full(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale,scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def offset_calculate_skip_step(self, latents, num_inner_steps, epsilon, guidance_scale,skip_step):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                if (i%skip_step)==0:
                    loss = latent_prev - latents_prev_rec
                else:
                    loss=torch.zeros_like(latent_prev)
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
    
    
    def invert_skip_step(self, image_gt, prompt, guidance_scale, skip_step,num_inner_steps=10, early_stop_epsilon=1e-5,scale=1.):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate_skip_step(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale,skip_step)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    
    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps
        
       