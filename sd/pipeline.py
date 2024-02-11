import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from PIL import Image


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt_t,
    uncond_prompt=None,
    input_image=None,
    latents_cus=None,
    strength=0.8,
    batch_size=5,
    do_cfg=True,
    art_lab=True,
    latents_custom=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            
            cond_context = prompt_t
            uncond_context = uncond_prompt
            # print("🫥"*40)
            # # Convert into a list of length Seq_Len=77
            # cond_tokens = tokenizer.batch_encode_plus(
            #     [prompt], padding="max_length", max_length=77
            # ).input_ids
            # # (Batch_Size, Seq_Len)
            # cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            # cond_context = clip(cond_tokens)
            # # Convert into a list of length Seq_Len=77
            # uncond_tokens = tokenizer.batch_encode_plus(
            #     [uncond_prompt], padding="max_length", max_length=77
            # ).input_ids
            # # (Batch_Size, Seq_Len)
            # uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            # uncond_context = clip(uncond_tokens)
            # # (Batch_Size, Seq_Len, Dim) 작 (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            
            
            # print(cond_context.size())
            # print(uncond_context.size())
            
            context = torch.cat([cond_context, uncond_context])
            
            
        elif art_lab:
            context = prompt_t
            # print("아트랩 시작")
        else:
            print("🤬"*40)
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt_t], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            print("😭"*30)
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device).to(device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            print("👑"*30)
            print("👑"*30)
            print("dfjskdfjksd")
            # print(input_image_tensor.device)
            # print(encoder_noise.device)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
            
        elif latents_custom:
            latents = latents_cus
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            timestep.to(device)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            
            # print("❄️"*40)
            # print(model_input.size())
            # print(context.size())
            # print("❄️"*40)
            
            
            model_output = diffusion(model_input, context, time_embedding)
            
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
                
           
            
            
            latents = latents.to('cuda')
            model_output = model_output.to('cuda')
    
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        # print(type(latents))
        # print(latents.size())
        # print("😊"*40)
        
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)
        
        # print(type(images))
        # print(images.size())
        # print("🦚"*40)

        # tensors = torch.chunk(images, chunks=5, dim=0)
        # for t in tensors:
        # print(t.size())  # 각각 torch.Size([1, 3, 512, 512]) 출력됨
        
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # print("🐬"*40)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
    
            # Image.fromarray(images)
        # print(images.shape)
        return images 
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)





