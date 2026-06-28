"""
扩图管线 - 使用 Stable Diffusion Inpainting/Outpainting
对生成的人脸图像进行向外扩展
工单编号：人工智能 NLP-Agent 数字人项目-文生图智能体任务
"""

import os
import torch
import logging
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler

# 设置环境变量禁用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger = logging.getLogger(__name__)


class OutpaintingPipeline:
    """
    扩图管线
    使用 Stable Diffusion Inpainting 实现图像向外扩展
    步骤：扩展画布 -> 创建 mask -> 局部重绘
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config["hardware"]["device"]
        # CPU 必须使用 float32
        if config["hardware"]["dtype"] == "fp16":
            logger.warning("CPU 不支持 fp16，自动切换到 fp32")
        self.dtype = torch.float32  # CPU 强制使用 float32
        self.gen_cfg = config["generation"]["outpainting"]

        self.pipe = None

        # 扩图提示词
        self.prompt = (
            "continuation of the same photo, same style, same lighting, "
            "seamless extension, high quality, detailed, natural background"
        )
        self.negative_prompt = (
            "ugly, blurry, low quality, distorted, deformed, "
            "disconnected, artificial edges, artifacts"
        )

    def load_models(self):
        """加载 Inpainting 模型到 CPU"""
        if self.pipe is not None:
            return

        logger.info("加载 Inpainting 模型到 CPU...")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # 移动到 CPU
        self.pipe = self.pipe.to("cpu")
        logger.info("Inpainting 模型已加载到 CPU")

        # CPU 优化配置
        hw = self.config["hardware"]
        if hw.get("enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
            logger.info("已启用注意力切片 (CPU 内存优化)")
        if hw.get("enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
            logger.info("已启用 VAE 切片 (CPU 内存优化)")
        if hw.get("enable_model_cpu_offload"):
            # CPU 不需要 offload，但保留选项
            pass

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # CPU 多线程优化
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        logger.info(f"PyTorch 线程数: {torch.get_num_threads()}")

        logger.info("Inpainting 模型加载完成")

    def outpaint(self, image: Image.Image, direction: str = "all") -> Image.Image:
        """
        扩图

        Args:
            image: 输入图像
            direction: 扩展方向 - "all", "right", "left", "top", "bottom"

        Returns:
            扩图后的图像
        """
        self.load_models()

        expand = self.gen_cfg["expand_pixels"]
        w, h = image.size

        # 根据方向确定新画布尺寸
        if direction == "all":
            new_w = w + expand * 2
            new_h = h + expand * 2
            offset_x, offset_y = expand, expand
        elif direction == "right":
            new_w = w + expand
            new_h = h
            offset_x, offset_y = 0, 0
        elif direction == "left":
            new_w = w + expand
            new_h = h
            offset_x, offset_y = expand, 0
        elif direction == "top":
            new_w = w
            new_h = h + expand
            offset_x, offset_y = 0, expand
        elif direction == "bottom":
            new_w = w
            new_h = h + expand
            offset_x, offset_y = 0, 0
        else:
            raise ValueError(f"不支持的方向: {direction}")

        # 创建扩展后的画布和 mask
        new_image = Image.new("RGB", (new_w, new_h), (127, 127, 127))
        new_image.paste(image, (offset_x, offset_y))

        mask = Image.new("L", (new_w, new_h), 0)

        if direction == "all":
            # 四周白色，中间黑色
            mask = Image.new("L", (new_w, new_h), 255)
            mask.paste(Image.new("L", (w, h), 0), (offset_x, offset_y))
        elif direction == "right":
            # 右侧白色条
            mask_pixels = mask.load()
            for x in range(new_w - expand, new_w):
                for y in range(new_h):
                    mask_pixels[x, y] = 255
        elif direction == "left":
            mask_pixels = mask.load()
            for x in range(expand):
                for y in range(new_h):
                    mask_pixels[x, y] = 255
        elif direction == "top":
            mask_pixels = mask.load()
            for x in range(new_w):
                for y in range(expand):
                    mask_pixels[x, y] = 255
        elif direction == "bottom":
            mask_pixels = mask.load()
            for x in range(new_w):
                for y in range(new_h - expand, new_h):
                    mask_pixels[x, y] = 255

        # 确保尺寸是 8 的倍数
        new_w_aligned = (new_w // 8) * 8
        new_h_aligned = (new_h // 8) * 8
        if new_w_aligned != new_w or new_h_aligned != new_h:
            new_image = new_image.resize((new_w_aligned, new_h_aligned), Image.LANCZOS)
            mask = mask.resize((new_w_aligned, new_h_aligned), Image.NEAREST)
            new_w, new_h = new_w_aligned, new_h_aligned

        # CPU 上的生成器
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.gen_cfg.get("seed", 42))

        logger.info(f"扩图: {direction} 方向，新尺寸 {new_w}x{new_h}... (CPU 推理)")

        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=new_image,
            mask_image=mask,
            num_inference_steps=self.gen_cfg["num_inference_steps"],
            guidance_scale=self.gen_cfg["guidance_scale"],
            generator=generator,
            output_type="pil",
        )

        logger.info(f"{direction} 方向扩图完成")
        return result.images[0]

    def outpaint_all(self, image: Image.Image) -> Image.Image:
        """四向扩图（一圈）"""
        current = image
        for direction in ["right", "left", "top", "bottom"]:
            logger.info(f"开始 {direction} 方向扩图...")
            current = self.outpaint(current, direction)
        logger.info("所有方向扩图完成")
        return current