"""
人脸旋转管线 - 使用 IP-Adapter FaceID + Stable Diffusion
生成右转、左转、端正三个角度的人脸图像
工单编号：人工智能 NLP-Agent 数字人项目-文生图智能体任务
"""

import os
import torch
import logging
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
import numpy as np

# 设置环境变量禁用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger = logging.getLogger(__name__)


class FaceRotationPipeline:
    """
    人脸旋转管线
    使用 IP-Adapter FaceID Plus v2 保持人脸身份特征
    通过文本提示控制生成角度
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config["hardware"]["device"]
        # CPU 必须使用 float32，不支持 float16
        if config["hardware"]["dtype"] == "fp16":
            logger.warning("CPU 不支持 fp16，自动切换到 fp32")
        self.dtype = torch.float32  # CPU 强制使用 float32
        self.model_cfg = config["model"]
        self.gen_cfg = config["generation"]["face_rotation"]

        # 懒惰加载
        self.pipe = None
        self.ip_adapter = None
        self.image_encoder = None

        # 各角度的提示词
        self.prompts = {
            "right": {
                "positive": "a photo of a face looking to the right, 3/4 profile view, turning head to the right side, well-lit, high quality, detailed skin texture",
                "negative": self.gen_cfg["negative_prompt"],
            },
            "left": {
                "positive": "a photo of a face looking to the left, 3/4 profile view, turning head to the left side, well-lit, high quality, detailed skin texture",
                "negative": self.gen_cfg["negative_prompt"],
            },
            "front": {
                "positive": "a photo of a face looking straight at the camera, front view, symmetrical face, well-lit, high quality, detailed skin texture",
                "negative": self.gen_cfg["negative_prompt"],
            },
        }

    def load_models(self):
        """加载所有模型（首次调用时加载）"""
        if self.pipe is not None:
            return

        model_name = self.model_cfg["base"]
        logger.info(f"加载基础模型: {model_name} (CPU 模式)")

        if model_name == "sd15":
            self._load_sd15()
        elif model_name == "sdxl":
            self._load_sdxl()
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")

        # CPU 优化配置 — 注意力切片已在 _load_sd15/_load_sdxl 中
        # 先于 load_ip_adapter 启用，避免覆盖 IP-Adapter 处理器
        hw = self.config["hardware"]
        if hw.get("enable_model_cpu_offload"):
            pass
        if hw.get("enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if hw.get("enable_vae_tiling"):
            self.pipe.enable_vae_tiling()

        # CPU 多线程优化
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        logger.info(f"PyTorch 线程数: {torch.get_num_threads()}")

        # 使用更快的调度器
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        logger.info(f"模型加载完成，设备: {self.device}")

    def _load_sd15(self):
        """加载 SD 1.5 模型 + IP-Adapter FaceID Plus v2 (CPU版本)"""
        cfg = self.model_cfg["sd15"]

        logger.info("正在加载 SD1.5 模型到 CPU...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            cfg["base_model"],
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # 移动到 CPU
        self.pipe = self.pipe.to("cpu")
        logger.info("模型已加载到 CPU")

        # 注意：注意力切片必须在 load_ip_adapter 之前启用，
        # 否则 SlicedAttnProcessor 会覆盖 IP-Adapter 的处理器，
        # 导致 encoder_hidden_states 的 tuple 格式无法被正确处理
        hw = self.config["hardware"]
        if hw.get("enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
            logger.info("已启用注意力切片 (先于 IP-Adapter)")
        if hw.get("enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
            logger.info("已启用 VAE 切片")
        if hw.get("enable_vae_tiling"):
            self.pipe.enable_vae_tiling()
            logger.info("已启用 VAE 平铺")

        # 加载 IP-Adapter
        logger.info("正在加载 IP-Adapter...")
        self.pipe.load_ip_adapter(
            cfg["ip_adapter"],
            subfolder=cfg["ip_adapter_subfolder"],
            weight_name=cfg["ip_adapter_weight_name"],
        )
        logger.info("IP-Adapter 加载完成")

    def _load_sdxl(self):
        """加载 SDXL 模型 + IP-Adapter FaceID Plus v2 (CPU版本)"""
        cfg = self.model_cfg["sdxl"]

        logger.info("正在加载 SDXL 模型到 CPU...")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg["base_model"],
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # 移动到 CPU
        self.pipe = self.pipe.to("cpu")
        logger.info("模型已加载到 CPU")

        # 注意力切片必须在 load_ip_adapter 之前启用
        hw = self.config["hardware"]
        if hw.get("enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
            logger.info("已启用注意力切片 (先于 IP-Adapter)")
        if hw.get("enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if hw.get("enable_vae_tiling"):
            self.pipe.enable_vae_tiling()

        self.pipe.load_ip_adapter(
            cfg["ip_adapter"],
            subfolder=cfg["ip_adapter_subfolder"],
            weight_name=cfg["ip_adapter_weight_name"],
        )

    def generate(self, face_image: Image.Image, angle: str) -> Image.Image:
        """
        生成指定角度的人脸图像

        Args:
            face_image: 输入的面部图像 (PIL Image)
            angle: 角度 - "right", "left", "front"

        Returns:
            生成的人脸图像
        """
        self.load_models()

        if angle not in self.prompts:
            raise ValueError(f"不支持的角度: {angle}，可选: {list(self.prompts.keys())}")

        prompt_data = self.prompts[angle]

        # 设置 IP-Adapter 权重
        self.pipe.set_ip_adapter_scale(self.gen_cfg["ip_adapter_scale"])

        # CPU 上的生成器（不需要 CUDA）
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.gen_cfg["seed"])

        logger.info(f"生成 {angle} 角度图像... (CPU 推理，可能需要较长时间)")

        # 生成（CPU 上推理）
        result = self.pipe(
            prompt=prompt_data["positive"],
            negative_prompt=prompt_data["negative"],
            ip_adapter_image=face_image,
            num_inference_steps=self.gen_cfg["num_inference_steps"],
            guidance_scale=self.gen_cfg["guidance_scale"],
            generator=generator,
            output_type="pil",
        )

        logger.info(f"{angle} 角度生成完成")
        return result.images[0]

    def generate_all_angles(self, face_image: Image.Image) -> dict:
        """
        生成所有三个角度的图像

        Returns:
            {"right": PIL.Image, "left": PIL.Image, "front": PIL.Image}
        """
        results = {}
        for angle in ["right", "front", "left"]:
            logger.info(f"开始生成 {angle} 角度...")
            results[angle] = self.generate(face_image, angle)
        return results