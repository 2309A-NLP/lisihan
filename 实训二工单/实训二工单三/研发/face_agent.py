"""
文生图智能体 - 核心智能体
协调人脸旋转 + 扩图的完整工作流（3D变形模式）
工单编号：人工智能 NLP-Agent 数字人项目-文生图智能体任务
"""

import os
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path
from PIL import Image

from pipelines.api_rotation import ApiRotationPipeline
from pipelines.outpainting import OutpaintingPipeline
from utils.face_utils import load_image, crop_face, resize_to_multiple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FaceAgent")


class FaceAgent:
    """
    文生图智能体
    功能：
    1. 输入面部图像，生成右转、左转、端正三个角度的人脸
    2. 对生成结果进行扩图
    """

    def __init__(self, config_path: str = "config.yaml"):
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = self.base_dir / "output"

        # 创建输出目录
        self.rotated_dir = self.output_dir / "rotated"
        self.outpainted_dir = self.output_dir / "outpainted"
        self.rotated_dir.mkdir(parents=True, exist_ok=True)
        self.outpainted_dir.mkdir(parents=True, exist_ok=True)

        # 懒加载管线
        self._face_rotation = None
        self._outpainting = None

        logger.info("=" * 60)
        logger.info("文生图智能体初始化完成")
#         logger.info(f"  基础模型: {self.config['model']['base']}")
        logger.info(f"  设备: {self.config['hardware']['device']}")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info("=" * 60)

    @property
    def face_rotation(self):
        if self._face_rotation is None:
            self._face_rotation = ApiRotationPipeline(self.config["model"])
        return self._face_rotation

    @property
    def outpainting(self):
        if self._outpainting is None:
            self._outpainting = OutpaintingPipeline(self.config)
        return self._outpainting

    def run(self, input_image_path: str = None, do_outpainting: bool = True,
            image_obj: Image.Image = None) -> dict:
        """
        执行完整工作流

        Args:
            input_image_path: 输入面部图像路径（和 image_obj 二选一）
            do_outpainting: 是否执行扩图
            image_obj: 直接传入 PIL Image 对象（和 input_image_path 二选一）

        Returns:
            {
                "input": 原图路径,
                "rotated": {"right": 路径, "left": 路径, "front": 路径},
                "outpainted": {"right": 路径, "left": 路径, "front": 路径} (可选)
            }
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 加载图像
        print("\n[步骤 1/4] 加载图像...")
        if image_obj is not None:
            face_image = image_obj.convert("RGB")
            print(f"  使用传入的图像对象")
        elif input_image_path:
            print(f"  输入: {input_image_path}")
            face_image = load_image(input_image_path)
        else:
            raise ValueError("必须提供 input_image_path 或 image_obj")
        face_image = resize_to_multiple(face_image)
        print(f"  图像尺寸: {face_image.size}")

        # 保存原图副本
        input_save_path = str(self.output_dir / f"input_{timestamp}.png")
        face_image.save(input_save_path)
        print(f"  原图已保存: {input_save_path}")

        # 2. 直接使用全图（API模式不做裁切，上下文信息有助于保持身份）
        print("\n[步骤 2/4] 图像准备完成（API模式，保留全图上下文）")
        processed_image = resize_to_multiple(face_image)
        print(f"  处理尺寸: {processed_image.size}")

        # 3. 生成三个角度
        print("\n[步骤 3/4] 生成三个角度的人脸...")
        print("  " + "-" * 40)
        results = self.face_rotation.generate_all_angles(processed_image)

        rotated_paths = {}
        for angle, img in results.items():
            save_path = str(self.rotated_dir / f"{angle}_{timestamp}.png")
            img.save(save_path)
            rotated_paths[angle] = save_path
            print(f"  [{angle}] 已保存: {save_path}")

        print("  " + "-" * 40)

        # 4. 扩图 — 优先使用 API（无需下载模型），失败则本地尝试
        outpainted_paths = {}
        if do_outpainting:
            print("\n[步骤 4/4] 执行扩图（API方式）...")
            print("  " + "-" * 40)
            for angle, img in results.items():
                print(f"  正在扩图 [{angle}]...", end=" ", flush=True)
                try:
                    # 先用 API 扩图（SiliconFlow，无需 GPU）
                    outpainted = self.face_rotation.outpaint_all(img)
                    print("API完成")
                except Exception as api_err:
                    # API 失败，尝试本地 SD Inpainting
                    print(f"\n  API扩图失败（{api_err}），尝试本地模型...")
                    try:
                        outpainted = self.outpainting.outpaint_all(img)
                        print("  本地扩图完成")
                    except Exception as local_err:
                        print(f"  本地扩图也失败: {local_err}，跳过此角度")
                        continue
                save_path = str(self.outpainted_dir / f"{angle}_outpainted_{timestamp}.png")
                outpainted.save(save_path)
                outpainted_paths[angle] = save_path
                print(f"  [{angle}] 扩图已保存: {save_path}")
            print("  " + "-" * 40)

        print(f"\n[文生图智能体] 全部完成！")
        print(f"  输出目录: {self.output_dir}")

        # 保存结果汇总
        summary_path = str(self.output_dir / f"summary_{timestamp}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"文生图智能体 - 处理结果\n")
            f.write(f"{'='*60}\n")
            f.write(f"输入图像: {input_image_path}\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("旋转结果:\n")
            for angle, path in rotated_paths.items():
                f.write(f"  {angle}: {path}\n")
            if outpainted_paths:
                f.write("\n扩图结果:\n")
                for angle, path in outpainted_paths.items():
                    f.write(f"  {angle}: {path}\n")
        print(f"  结果汇总: {summary_path}")

        return {
            "input": input_save_path,
            "rotated": rotated_paths,
            "outpainted": outpainted_paths,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="文生图智能体 - 人脸旋转 + 扩图"
    )
    parser.add_argument("input", type=str, help="输入面部图像路径")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--no-outpainting", action="store_true", help="跳过扩图步骤"
    )
    parser.add_argument(
        "--crop-only", action="store_true", help="只做人脸裁剪，不做生成（测试用）"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return

    agent = FaceAgent(args.config)
    agent.run(args.input, do_outpainting=not args.no_outpainting)


if __name__ == "__main__":
    main()
