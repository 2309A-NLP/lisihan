"""
文生图智能体 - Gradio Web UI (重设计版)
上传照片后能清晰看到原图和生成结果
工单编号：人工智能 NLP-Agent 数字人项目-文生图智能体任务
"""

import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 兼容性补丁
import jinja2
jinja2.environment.Environment._load_template = lambda self, name, globals: (
    self.loader.load(self, name, self.make_globals(globals))
    if self.loader is not None
    else (_ for _ in ()).throw(__import__('jinja2').TemplateNotFound(name))
)

import gradio as gr
from face_agent import FaceAgent
from PIL import Image
import numpy as np

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        cfg = os.path.join(os.path.dirname(__file__), "config.yaml")
        _agent = FaceAgent(cfg)
    return _agent

def update_preview(input_image):
    """上传后立即更新预览"""
    if input_image is None:
        return None
    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
    else:
        img = input_image
    preview = img.copy()
    preview.thumbnail((400, 400))
    return preview

def process_image(input_image, do_outpainting, progress=gr.Progress()):
    if input_image is None:
        return [None]*4 + ["请先上传一张面部照片"]

    progress(0, desc="准备图像...")
    temp_dir = os.path.join(os.path.dirname(__file__), "input")
    os.makedirs(temp_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = os.path.join(temp_dir, f"upload_{ts}.png")

    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
    else:
        img = input_image
    img.save(temp_path)

    # 生成预览缩略图
    preview = img.copy()
    preview.thumbnail((400, 400))

    try:
        progress(0.2, desc="加载模型中...")
        agent = get_agent()

        progress(0.3, desc="生成三个角度...")
        result = agent.run(temp_path, do_outpainting=bool(do_outpainting))

        right_img = Image.open(result["rotated"]["right"])
        front_img = Image.open(result["rotated"]["front"])
        left_img = Image.open(result["rotated"]["left"])

        progress(0.8, desc="扩图中...")
        outpainted_img = None
        if do_outpainting and result.get("outpainted"):
            out_right = Image.open(result["outpainted"]["right"]).copy()
            out_front = Image.open(result["outpainted"]["front"]).copy()
            out_left = Image.open(result["outpainted"]["left"]).copy()
            w, h = out_right.size
            canvas = Image.new("RGB", (w * 3, h))
            canvas.paste(out_right, (0, 0))
            canvas.paste(out_front, (w, 0))
            canvas.paste(out_left, (w * 2, 0))
            outpainted_img = canvas

        progress(1.0, desc="完成!")
        return [right_img, front_img, left_img, outpainted_img, "处理完成！"]

    except Exception as e:
        return [None, None, None, None, f"错误: {str(e)}"]

CSS = """
body { background: #0f0f13; }
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
.title-box { text-align: center; padding: 24px 0 12px; }
.title-box h1 { font-size: 28px; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.title-box p { color: #999; font-size: 14px; margin-top: 6px; }
.section-title { font-size: 16px; font-weight: 600; color: #ccc; margin-bottom: 8px; }
.preview-box { border: 2px solid #334; border-radius: 12px; padding: 6px; background: #16161f; }
.result-card { border: 2px solid #334; border-radius: 12px; padding: 6px; background: #16161f; }
.result-card img { width: 100%; object-fit: contain; }
.angle-label { text-align: center; font-size: 14px; font-weight: 500; color: #aab; margin: 4px 0 2px; }
footer { display: none !important; }
#input_image { min-height: 340px; }
"""

with gr.Blocks(css=CSS, title="文生图智能体", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="title-box">
      <h1>文生图智能体</h1>
      <p>上传一张面部照片 → 自动生成右转 · 端正 · 左转三个角度 + 可选扩图</p>
    </div>
    """)

    # ===== 第一行：上传 + 原图预览 =====
    with gr.Row(equal_height=True):
        with gr.Column(scale=3, min_width=300):
            gr.HTML('<div class="section-title">上传面部照片</div>')
            input_image = gr.Image(label="", type="pil", height=340)

        with gr.Column(scale=4, min_width=400):
            gr.HTML('<div class="section-title">原图预览</div>')
            preview_img = gr.Image(label="", type="pil", height=340, elem_classes="preview-box")

    # ===== 选项 + 按钮 =====
    with gr.Row():
        with gr.Column(scale=1, min_width=120):
            outpaint_check = gr.Checkbox(label="执行扩图", value=True)
        with gr.Column(scale=4):
            submit_btn = gr.Button("开始生成", variant="primary", size="lg")
        with gr.Column(scale=3):
            status = gr.Textbox(label="状态", interactive=False)

    gr.Markdown("---")

    # ===== 生成结果 =====
    gr.HTML('<div class="section-title">生成结果</div>')
    with gr.Row(equal_height=True):
        with gr.Column(min_width=180):
            gr.HTML('<div class="angle-label">左转</div>')
            left_img = gr.Image(label="", type="pil", height=260, elem_classes="result-card")
        with gr.Column(min_width=180):
            gr.HTML('<div class="angle-label">端正</div>')
            front_img = gr.Image(label="", type="pil", height=260, elem_classes="result-card")
        with gr.Column(min_width=180):
            gr.HTML('<div class="angle-label">右转</div>')
            right_img = gr.Image(label="", type="pil", height=260, elem_classes="result-card")

    # ===== 扩图结果 =====
    with gr.Row():
        outpainted_img = gr.Image(label="扩图结果（三张拼接）", type="pil", elem_classes="result-card")

    # ===== 事件绑定 =====
    # 上传照片后立即更新预览
    input_image.change(
        fn=update_preview,
        inputs=[input_image],
        outputs=[preview_img],
    )

    # 点击生成
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, outpaint_check],
        outputs=[right_img, front_img, left_img, outpainted_img, status],
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

