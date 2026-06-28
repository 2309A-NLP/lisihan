"""
人脸处理工具函数
工单编号：人工智能 NLP-Agent 数字人项目-文生图智能体任务
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Image.Image:
    """加载图像，统一转为 RGB"""
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    else:
        raise TypeError(f"不支持的图像类型: {type(image_path)}")
    return img


def resize_to_multiple(image: Image.Image, multiple: int = 8) -> Image.Image:
    """将图像尺寸调整为 multiple 的倍数（SD 要求）"""
    w, h = image.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def crop_face(image: Image.Image, face_bbox=None, margin: float = 0.3) -> Image.Image:
    """
    裁切人脸区域，带边距
    face_bbox: (x1, y1, x2, y2) 或 None（自动检测）
    margin: 边距比例，0.3 表示向外扩展 30%
    """
    if face_bbox is None:
        face_bbox = detect_face(image)

    if face_bbox is None:
        logger.warning("未检测到人脸，返回原图")
        return image

    x1, y1, x2, y2 = face_bbox
    w, h = image.size

    # 计算边距
    face_w = x2 - x1
    face_h = y2 - y1
    margin_x = int(face_w * margin)
    margin_y = int(face_h * margin)

    # 扩展边界
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    return image.crop((x1, y1, x2, y2))


def detect_face(image: Image.Image) -> tuple | None:
    """
    使用 OpenCV Haar Cascade 检测人脸
    返回 (x1, y1, x2, y2) 或 None
    """
    # 转为 OpenCV 格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    if len(faces) == 0:
        return None

    # 取最大的人脸
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, x + w, y + h)


def create_outpaint_mask(
    image: Image.Image, expand_pixels: int = 128
) -> Image.Image:
    """
    创建扩图用的 mask
    白色区域（255）= 需要生成的部分
    黑色区域（0）= 保留原图
    """
    w, h = image.size
    mask_w = w + expand_pixels * 2
    mask_h = h + expand_pixels * 2

    mask = np.ones((mask_h, mask_w), dtype=np.uint8) * 255

    # 中间原始图像区域设为黑色（保留）
    margin = expand_pixels
    mask[margin : margin + h, margin : margin + w] = 0

    return Image.fromarray(mask, mode="L")


def expand_canvas(
    image: Image.Image, expand_pixels: int = 128, fill_color=(127, 127, 127)
) -> Image.Image:
    """扩展画布，四周填充灰色"""
    w, h = image.size
    new_w = w + expand_pixels * 2
    new_h = h + expand_pixels * 2

    new_img = Image.new("RGB", (new_w, new_h), fill_color)
    new_img.paste(image, (expand_pixels, expand_pixels))
    return new_img


def crop_center(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """中心裁剪到目标尺寸"""
    w, h = image.size
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return image.crop((left, top, right, bottom))
