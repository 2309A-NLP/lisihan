"""
3D人脸旋转管线
使用 insightface 的 2d106det 模型检测106个人脸关键点
通过3D旋转 + Delaunay三角剖分 + 仿射变换实现人脸角度转向

流程:
  1. insightface 2d106det 检测106个2D人脸关键点
  2. 估算深度信息构建3D关键点
  3. 对关键点绕Y轴旋转（yaw）实现左右转头
  4. Delaunay三角剖分
  5. 逐三角形应用仿射变换渲染旋转后图像
  6. 对空洞区域做修复

依赖: insightface, opencv-python, numpy, pillow
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ============================================================
# 人脸旋转角度定义
# ============================================================
ANGLES = {
    "right": {"yaw": 30, "pitch": 0, "roll": 0},
    "front": {"yaw": 0, "pitch": 0, "roll": 0},
    "left": {"yaw": -30, "pitch": 0, "roll": 0},
}


class Face3DRotationPipeline:
    """基于3D人脸关键点的旋转管线（纯CPU，无需额外模型下载）"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._landmark_detector = None

    # ----------------------------------------------------------
    # 初始化关键点检测器
    # ----------------------------------------------------------
    def _ensure_initialized(self):
        if self._landmark_detector is not None:
            return

        logger.info("初始化 insightface 关键点检测器 (2d106det)...")
        try:
            import insightface
            from insightface.model_zoo import get_model

            # 2d106det 模型会从 insightface 服务器自动下载（国内可访问）
            self._landmark_detector = get_model('2d106det')
            self._landmark_detector.prepare(ctx_id=-1)  # -1 = CPU
            logger.info("insightface 2d106det 初始化完成")

        except Exception as e:
            # 如果 2d106det 不可用，回退到 RetinaFace 的5点检测
            logger.warning(f"2d106det 加载失败 ({e})，回退到 RetinaFace 5点检测")
            try:
                from insightface.app import FaceAnalysis
                self._app = FaceAnalysis(name='buffalo_l')
                self._app.prepare(ctx_id=-1)
                self._use_5point = True
                logger.info("RetinaFace 5点检测回退模式")
            except Exception as e2:
                raise RuntimeError(
                    f"insightface 初始化失败 (2d106det: {e}, RetinaFace: {e2})"
                )

    # ----------------------------------------------------------
    # 检测人脸关键点
    # ----------------------------------------------------------
    def _detect_landmarks(self, img_bgr: np.ndarray):
        """
        检测人脸关键点

        Returns:
            (points_2d, points_3d): (N, 2) 和 (N, 3)，N=106 或 5
            或 None（未检测到人脸）
        """
        self._ensure_initialized()
        h, w = img_bgr.shape[:2]

        if not getattr(self, '_use_5point', False):
            # ---- 106点模式 ----
            landmarks = self._landmark_detector.get(img_bgr)
            if landmarks is None or len(landmarks) == 0:
                return None

            # 取最大人脸
            if len(landmarks) > 1:
                # 按边界框面积排序取最大
                areas = []
                for lm in landmarks:
                    xs, ys = lm[:, 0], lm[:, 1]
                    areas.append((xs.max() - xs.min()) * (ys.max() - ys.min()))
                landmarks = landmarks[np.argmax(areas)]

            pts = np.array(landmarks, dtype=np.float32)  # (106, 2)

            # 估算深度（z坐标）
            # 使用标准人脸深度模型：鼻子附近z大，边缘z小
            face_center = pts.mean(axis=0)
            distances = np.linalg.norm(pts - face_center, axis=1)
            max_dist = distances.max() if distances.max() > 0 else 1

            # z值与到中心的距离负相关（边缘更靠后），加上面部先验
            z_vals = -distances * 0.15  # 简单深度估算
            # 鼻尖区域突出
            nose_idx = 52  # 106点模型中鼻尖大致索引
            if nose_idx < len(pts):
                z_vals[nose_idx] += 15.0  # 鼻尖突出

            pts_3d = np.column_stack([pts, z_vals])
            return pts, pts_3d

        else:
            # ---- FaceAnalysis模式 ----
            faces = self._app.get(img_bgr)
            if not faces:
                return None
            face = faces[0]

            # 方案A: 使用3D 68点关键点（带真深度，最佳）
            if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                pts = np.array(face.landmark_3d_68, dtype=np.float32)  # (68, 3)
                pts_2d = pts[:, :2].copy()
                pts_3d = pts[:, :3].copy()
                return pts_2d, pts_3d

            # 方案B: 使用2D 106点关键点+估算深度
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                pts = np.array(face.landmark_2d_106, dtype=np.float32)
                pts_2d = pts[:, :2]
                if pts.shape[1] >= 3:
                    pts_3d = pts[:, :3].copy()
                    pts_3d[:, 2] = 0
                else:
                    pts_3d = np.column_stack([pts_2d, np.zeros(len(pts), dtype=np.float32)])
                face_center = pts_2d.mean(axis=0)
                dists = np.linalg.norm(pts_2d - face_center, axis=1)
                pts_3d[:, 2] = -dists * 0.15
                if len(pts) > 52:
                    pts_3d[52, 2] = 15.0
                return pts_2d, pts_3d

            # 方案C: 回退到5点+轮廓点
            kps = face.kps
            extra_points = self._generate_extra_points(kps, face.bbox, h, w)
            all_pts = np.vstack([kps, extra_points]).astype(np.float32)
            face_center = all_pts.mean(axis=0)
            distances = np.linalg.norm(all_pts - face_center, axis=1)
            z_vals = -distances * 0.15
            z_vals[2] += 15.0
            pts_3d = np.column_stack([all_pts, z_vals])
            return all_pts, pts_3d

    # ----------------------------------------------------------
    # 5点模式下补全更多关键点
    # ----------------------------------------------------------
    def _generate_extra_points(self, kps, bbox, h, w):
        """基于5个关键点 + 人脸边界框生成额外轮廓点"""
        points = []
        x1, y1, x2, y2 = bbox

        # 人脸矩形四角
        margin = int((x2 - x1) * 0.1)
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)

        # 在面部轮廓上采样点（沿椭圆）
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx, ry = (x2 - x1) / 2, (y2 - y1) / 2

        for angle in np.linspace(0, 2 * np.pi, 16):
            px = cx + rx * np.cos(angle)
            py = cy + ry * np.sin(angle)
            points.append([px, py])

        # 在额头和下巴中间添加点
        # 连接眉毛到下巴的弧线
        chin_y = y2
        forehead_y = y1
        for frac in [0.25, 0.5, 0.75]:
            px = x1 + frac * (x2 - x1)
            points.append([px, chin_y])
            points.append([px, forehead_y])

        return np.array(points, dtype=np.float32)

    # ----------------------------------------------------------
    # 3D旋转人脸
    # ----------------------------------------------------------
    def rotate_face(self, image: Image.Image, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
        """
        3D人脸旋转

        Args:
            image: 输入人脸图像 (PIL RGB)
            yaw_deg: 水平旋转角（度），负=左转，正=右转
            pitch_deg: 俯仰角
            roll_deg: 旋转角

        Returns:
            旋转后人脸图像 (PIL RGB)
        """
        img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # 检测关键点
        result = self._detect_landmarks(img_bgr)
        if result is None:
            logger.warning("未检测到人脸，返回原图")
            return image

        pts_2d, pts_3d = result
        return self._warp_face(img_bgr, pts_2d, pts_3d, yaw_deg, pitch_deg, roll_deg)

    # ----------------------------------------------------------
    # 核心：3D旋转 + Delaunay三角剖分 + 仿射变换
    # ----------------------------------------------------------
    def _warp_face(self, img_bgr, pts_2d, pts_3d, yaw_deg, pitch_deg, roll_deg):
        h, w = img_bgr.shape[:2]

        # ---- 1. 构建3D旋转矩阵 ----
        center_3d = pts_3d.mean(axis=0)

        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ])
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1],
        ])
        R = Rz @ Ry @ Rx

        # ---- 2. 旋转3D关键点 ----
        rotated_3d = (pts_3d - center_3d) @ R.T + center_3d
        rotated_2d = rotated_3d[:, :2].astype(np.float32)

        # ---- 3. 添加边界锚点，使三角剖分覆盖全图 ----
        # 边界锚点不旋转（或极小位移），让旋转从人脸平滑过渡到背景
        h, w = img_bgr.shape[:2]
        margin = 5
        boundary_pts = []
        # 四边各均匀加8个点
        for x in np.linspace(margin, w - margin, 10):
            boundary_pts.append([x, margin])           # 上边
            boundary_pts.append([x, h - margin])       # 下边
        for y in np.linspace(margin, h - margin, 8):
            boundary_pts.append([margin, y])           # 左边
            boundary_pts.append([w - margin, y])       # 右边
        # 四角
        boundary_pts.append([margin, margin])
        boundary_pts.append([w - margin, margin])
        boundary_pts.append([margin, h - margin])
        boundary_pts.append([w - margin, h - margin])

        boundary_pts = np.array(boundary_pts, dtype=np.float32)

        # 合并人脸关键点 + 边界锚点
        all_src = np.vstack([pts_2d, boundary_pts])  # (N + B, 2)
        all_dst = np.vstack([rotated_2d, boundary_pts])  # 边界锚点不动

        # ---- 4. Delaunay三角剖分（覆盖全图） ----
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        for p in all_src:
            subdiv.insert((int(p[0]), int(p[1])))

        triangles = subdiv.getTriangleList()
        valid_tris = []
        for t in triangles:
            p1 = (int(t[0]), int(t[1]))
            p2 = (int(t[2]), int(t[3]))
            p3 = (int(t[4]), int(t[5]))
            # 只要三角形有一部分在画面内即可
            cx = (p1[0] + p2[0] + p3[0]) / 3
            cy = (p1[1] + p2[1] + p3[1]) / 3
            if 0 <= cx < w and 0 <= cy < h:
                valid_tris.append((p1, p2, p3))

        # ---- 4. 三角仿射变换渲染 ----
        output = np.zeros((h, w, 3), dtype=img_bgr.dtype)
        coverage = np.zeros((h, w), dtype=np.uint8)

        for src_p1, src_p2, src_p3 in valid_tris:
            src_tri = np.array([src_p1, src_p2, src_p3], dtype=np.float32)

            # 在合并后的点集(all_src)中找到对应索引，取all_dst中的目标位置
            dst_tri = np.zeros((3, 2), dtype=np.float32)
            for i, sp in enumerate([src_p1, src_p2, src_p3]):
                dists = np.sum((all_src - sp) ** 2, axis=1)
                idx = np.argmin(dists)
                dst_tri[i] = all_dst[idx]

            # 跳过无效三角形
            if cv2.contourArea(src_tri) < 1 or cv2.contourArea(dst_tri) < 1:
                continue

            # 计算仿射变换
            affine_mat = cv2.getAffineTransform(src_tri, dst_tri)

            # Warp整图（比ROI省事，mask约束输出区域）
            warped = cv2.warpAffine(
                img_bgr, affine_mat, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # 目标三角形mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_tri.astype(np.int32), 255)

            # 写回输出
            np.copyto(output, warped, where=(mask > 0)[:, :, None])
            coverage = cv2.bitwise_or(coverage, mask)

        # ---- 5. 修复空洞 ----
        holes = cv2.bitwise_not(coverage)
        # 只修复人脸附近的空洞
        kernel = np.ones((7, 7), np.uint8)
        face_region = cv2.dilate(coverage, kernel, iterations=2)
        repair_region = cv2.bitwise_and(holes, face_region)

        if np.any(repair_region):
            output = cv2.inpaint(output, repair_region, 5, cv2.INPAINT_TELEA)

        logger.info(f"渲染完成: {len(valid_tris)} 三角形")

        return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    # ----------------------------------------------------------
    # 生成三个标准角度
    # ----------------------------------------------------------
    def generate_all_angles(self, face_image: Image.Image) -> dict:
        """生成三个角度的人脸图像"""
        results = {}
        for angle_name, params in ANGLES.items():
            logger.info(f"生成 {angle_name} 角度 (yaw={params['yaw']}°)...")
            results[angle_name] = self.rotate_face(
                face_image,
                yaw_deg=params["yaw"],
                pitch_deg=params["pitch"],
                roll_deg=params["roll"],
            )
        return results
