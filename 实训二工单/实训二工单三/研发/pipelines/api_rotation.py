"""
API人脸旋转管线
使用 SiliconFlow 的 Qwen/Qwen-Image-Edit 模型
通过云端API调用实现人脸角度旋转，保留身份特征

依赖: requests (可选), urllib, json, base64
"""

import base64
import json
import logging
import urllib.request
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 各角度提示词
ANGLES = {
    "right": {
        "prompt": "拍照视角向右转30度：人物头部向左转，右侧脸颊变小，看到左侧脸颊和左耳更多，完全同一个人，不改变五官长相",
        "negative_prompt": "不同的人，改变长相，正面视角，美颜，磨皮，滤镜，修改五官，扭曲",
        "strength": 0.5,
        "mirror": False,
    },
    "front": {
        "prompt": "正面拍照视角：人物正对镜头，面部左右对称，五官清晰，完全同一个人，不改变长相",
        "negative_prompt": "不同的人，改变长相，侧脸，美颜，磨皮，滤镜，修改五官",
        "strength": 0.3,
        "mirror": False,
    },
    "left": {
        "prompt": "拍照视角向左转30度：人物头部向右转，左侧脸颊变小，看到右侧脸颊和右耳更多，完全同一个人，不改变五官长相",
        "negative_prompt": "不同的人，改变长相，正面视角，美颜，磨皮，滤镜，修改五官，扭曲",
        "strength": 0.5,
        "mirror": False,
    },
}


class ApiRotationPipeline:
    """基于火山引擎 Doubao 的人脸旋转管线（CPU可用，调用云端GPU）"""

    def __init__(self, config: dict):
        self.config = config
        self.api_key = os.environ.get(
            "ARK_API_KEY",
            "ark-7419c003-7697-4d54-bbec-f08ad08094e5-25ec6",
        )
        self.endpoint_id = os.environ.get(
            "ARK_ENDPOINT_ID",
            "ep-20260625141102-6t8th",
        )
        self.base_url = os.environ.get(
            "ARK_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3",
        )
        self.image_size = 1920
        self.model = config.get("base", "sd15")

        if not self.api_key:
            raise ValueError(
                "未设置 API Key！请在环境变量 ARK_API_KEY 中设置"
            )

        logger.info(f"API旋转管线初始化: endpoint={self.endpoint_id}, size={self.image_size}")

    # ----------------------------------------------------------
    # 编码图片为 base64 data URI
    # ----------------------------------------------------------
    @staticmethod
    def _image_to_data_uri(image: Image.Image) -> str:
        """PIL Image -> data:image/png;base64,..."""
        # 确保是正方形
        w, h = image.size
        if w != h:
            side = max(w, h)
            square = Image.new("RGB", (side, side), (127, 127, 127))
            offset_x = (side - w) // 2
            offset_y = (side - h) // 2
            square.paste(image, (offset_x, offset_y))
            image = square

        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    # ----------------------------------------------------------
    # 调用 API 生成
    # ----------------------------------------------------------
    def _call_api(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.3,
    ) -> Image.Image:
        """调用火山引擎 Doubao API 生成旋转后图像"""
        s = self.image_size
        img_resized = image.resize((s, s), Image.LANCZOS)
        b64_data = self._image_to_data_uri(img_resized)

        payload = {
            "model": self.endpoint_id,
            "prompt": prompt,
            "image": b64_data,
            "size": f"{s}x{s}",
            "n": 1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/images/generations"

        logger.info(f"调用 Doubao: {self.endpoint_id} (size={s})")
        logger.debug(f"  prompt: {prompt[:60]}...")

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                img_url = result["data"][0]["url"]
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(
                f"API 请求失败 (HTTP {e.code}): {body[:300]}"
            )
        except Exception as e:
            raise RuntimeError(f"API 请求异常: {e}")

        # 下载结果
        try:
            with urllib.request.urlopen(img_url, timeout=60) as resp:
                img_data = resp.read()
            result_img = Image.open(BytesIO(img_data)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"下载结果图片失败: {e}")

        logger.info("API 生成完成")
        return result_img

    # ----------------------------------------------------------
    # 单一角度旋转
    # ----------------------------------------------------------
    def rotate_face(
        self,
        image: Image.Image,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> Image.Image:
        """
        旋转人脸到指定角度

        注意: API 模式通过文本 prompt 控制角度，yaw_deg/pitch_deg/roll_deg 仅作为参考。
        实际角度控制由 Doubao 模型根据 prompt 理解实现。
        """
        # 根据角度选择最接近的预定义 prompt
        if abs(yaw_deg) <= 15:
            angle_name = "front"
        elif yaw_deg > 0:
            angle_name = "right"
        else:
            angle_name = "left"

        angle_cfg = ANGLES[angle_name]
        return self._call_api(
            image,
            prompt=angle_cfg["prompt"],
            negative_prompt=angle_cfg["negative_prompt"],
        )

    # ----------------------------------------------------------
    # API 扩图（outpainting）
    # ----------------------------------------------------------
    def outpaint_all(self, image: Image.Image) -> Image.Image:
        """
        使用 API 进行四向扩图（一圈）
        通过扩展画布+mask 控制，API 自动填充扩展区域
        """
        expand_pixels = self.config.get("generation", {}).get("outpainting", {}).get("expand_pixels", 128)
        w, h = image.size

        # 扩大 20% 画布（太小的话 API 可能无法正确区分边界）
        if w < 200 or h < 200:
            expand = 200
        else:
            expand = max(expand_pixels, int(min(w, h) * 0.25))

        new_w = w + expand * 2
        new_h = h + expand * 2

        # 对齐到 8 的倍数
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        offset_x, offset_y = (new_w - w) // 2, (new_h - h) // 2

        # 创建扩展后的画布（灰色填充）
        new_img = Image.new("RGB", (new_w, new_h), (127, 127, 127))
        new_img.paste(image, (offset_x, offset_y))

        # 创建 mask：白色=重新生成区域，黑色=保留原图区域
        mask = Image.new("L", (new_w, new_h), 255)
        mask.paste(Image.new("L", (w, h), 0), (offset_x, offset_y))

        logger.info(f"API 扩图: {new_w}x{new_h} (expand={expand})")

        prompt = (
            "向外扩展这张图像，保持原始图像的风格、颜色、光照、纹理完全一致，"
            "扩展部分与原始图像无缝衔接，自然连续，看不出拼接痕迹"
        )
        negative_prompt = "条纹、拼接痕迹、颜色突变、内容不一致、扭曲、模糊"

        # 调用 API 进行 image-to-image 编辑
        return self._call_api_image_edit(
            image=new_img,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

    # ----------------------------------------------------------
    # 带 mask 的 API 调用（用于扩图/修图）
    # ----------------------------------------------------------
    def _call_api_image_edit(
        self,
        image: Image.Image,
        mask: Image.Image = None,
        prompt: str = "",
        negative_prompt: str = "",
    ) -> Image.Image:
        """调用 Doubao API 进行带 mask 的图像编辑"""
        b64_data = self._image_to_data_uri(image)

        payload = {
            "model": self.endpoint_id,
            "prompt": prompt,
            "image": b64_data,
            "size": f"{self.image_size}x{self.image_size}",
            "n": 1,
        }

        # 添加 mask（如果有）
        if mask is not None:
            payload["mask"] = self._image_to_data_uri(mask)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/images/generations"

        logger.info(f"调用 Doubao 图像编辑: {self.endpoint_id}")

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())
                img_url = result["data"][0]["url"]
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(
                f"API 图像编辑失败 (HTTP {e.code}): {body[:300]}"
            )
        except Exception as e:
            raise RuntimeError(f"API 图像编辑异常: {e}")

        # 下载结果
        try:
            with urllib.request.urlopen(img_url, timeout=60) as resp:
                img_data = resp.read()
            result_img = Image.open(BytesIO(img_data)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"下载结果图片失败: {e}")

        logger.info("API 图像编辑完成")
        return result_img

    def generate_all_angles(self, face_image: Image.Image) -> dict:
        """生成右转/正面/左转 三个角度"""
        results = {}
        for angle_name, angle_cfg in ANGLES.items():
            logger.info(f"生成 {angle_name} 角度...")

            img = face_image
            # 右转用镜像法：先镜像→左转→再翻回
            if angle_cfg.get("mirror", False):
                img = face_image.transpose(Image.FLIP_LEFT_RIGHT)

            result = self._call_api(
                img,
                prompt=angle_cfg["prompt"],
                negative_prompt=angle_cfg["negative_prompt"],
                strength=angle_cfg.get("strength", 0.3),
            )

            # 镜像法结果再翻回来
            if angle_cfg.get("mirror", False):
                result = result.transpose(Image.FLIP_LEFT_RIGHT)

            results[angle_name] = result
        return results
