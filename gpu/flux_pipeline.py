#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUX.1-dev 模型自定义管道
"""

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

class FluxPipeline(DiffusionPipeline):
    """
    FLUX.1-dev 模型的自定义管道类
    """
    def __init__(self, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None, 
                 safety_checker=None, feature_extractor=None, requires_safety_checker=False,
                 text_encoder_2=None, tokenizer_2=None, transformer=None):
        super().__init__()
        
        # 保存所有组件
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        **kwargs
    ):
        # 处理提示词
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` 必须是字符串或字符串列表，但收到了 {type(prompt)}")
            
        # 处理负面提示词
        if negative_prompt is None:
            negative_prompt = ""
            
        # 编码提示词
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # 获取文本嵌入
        text_embeddings = self.text_encoder(text_input_ids)[0]
        
        # 编码负面提示词
        uncond_input = self.tokenizer(
            [negative_prompt] * batch_size,
            padding="max_length",
            max_length=text_input_ids.shape[-1],
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # 复制以匹配每个提示词的图像数量
        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_embeddings = uncond_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
        
        # 连接用于分类器自由引导
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # 准备潜在向量
        latents_shape = (batch_size * num_images_per_prompt, self.unet.config.in_channels, height // 8, width // 8)
        
        if latents is None:
            if generator is None:
                generator = torch.Generator(device=self.device).manual_seed(0)
            latents = torch.randn(latents_shape, generator=generator, device=self.device)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"潜在向量形状不匹配，收到 {latents.shape}，期望 {latents_shape}")
                
        # 设置时间步
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 缩放潜在向量
        latents = latents * self.scheduler.init_noise_sigma
        
        # 去噪循环
        for i, t in enumerate(self.scheduler.timesteps):
            # 扩展潜在向量用于分类器自由引导
            latent_model_input = torch.cat([latents] * 2)
            
            # 预测噪声残差
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # 执行引导
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 计算前一个噪声样本
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # 回调
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
                
        # 缩放并解码潜在向量
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        
        # 转换为RGB
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        # 转换为输出格式
        if output_type == "pil":
            from PIL import Image
            
            images = []
            for i, img in enumerate(image):
                img = (img * 255).round().astype("uint8")
                images.append(Image.fromarray(img))
        else:
            images = image
            
        if not return_dict:
            return images
            
        return StableDiffusionPipelineOutput(images=images)
