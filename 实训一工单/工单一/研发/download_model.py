import os
import urllib.request
import ssl

# 禁用 SSL 验证（仅用于下载）
ssl._create_default_https_context = ssl._create_unverified_context

# 目标保存路径
model_dir = r"C:\Users\freedom\.conda\envs\py310\Lib\site-packages\ultralytics\cfg\models\v8"
model_path = os.path.join(model_dir, "doclayout_yolo_docstructbench_imgsz1280_2501.pt")

os.makedirs(model_dir, exist_ok=True)

# 备用下载链接 (GitHub 镜像)
urls = [
    "https://github.com/opendatalab/PDF-Extract-Kit/raw/main/models/Layout/doclayout_yolo_docstructbench_imgsz1280_2501.pt",  # GitHub RAW (可能需要代理)
    "https://cdn.hf-mirror.com/opendatalab/PDF-Extract-Kit/resolve/main/models/Layout/doclayout_yolo_docstructbench_imgsz1280_2501.pt", # 另一个镜像尝试
]

for url in urls:
    print(f"尝试从 {url} 下载...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✅ 模型下载成功！已保存至 {model_path}")
        break
    except Exception as e:
        print(f"下载失败: {e}")
        continue
else:
    print("\n❌ 所有自动下载方式均失败。")
    print("请手动从 ModelScope 下载：")
    print("1. 访问 https://modelscope.cn/models/opendatalab/PDF-Extract-Kit/files")
    print("2. 进入 models/Layout/ 目录")
    print("3. 下载 doclayout_yolo_docstructbench_imgsz1280_2501.pt")
    print(f"4. 放到 {model_dir}")