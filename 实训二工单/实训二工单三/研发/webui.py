# -*- coding: utf-8 -*-

"""
文生图智能体 - 现代化Web界面（纯Python标准库，无需额外依赖）
包含：上传预览、结果展示、历史记录
工单编号：人工智能 NLP-Agent 数字人项目-文生图智能体任务
"""

import sys
import os
import json
import base64
import io
import threading
import webbrowser
import datetime
import traceback

# ===== 修复 Windows 控制台编码问题 =====
if sys.platform == "win32":
    import io as _io

    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from PIL import Image
from face_agent import FaceAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
_CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = FaceAgent(_CONFIG_PATH)
    return _agent


# ========== 历史记录管理 ==========

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(record):
    history = load_history()
    history.insert(0, record)
    if len(history) > 50:
        history = history[:50]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ========== HTML 页面模板 ==========

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>文生图智能体</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Microsoft YaHei','PingFang SC',sans-serif;background:#f0f2f5;padding:20px;color:#333}
.container{max-width:1100px;margin:0 auto}
header{text-align:center;padding:20px 0 10px}
header h1{font-size:26px;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
header p{color:#888;font-size:13px;margin-top:4px}
.card{background:#fff;border-radius:12px;padding:24px;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08)}

/* 上传区 */
.upload-area{display:flex;gap:24px;align-items:flex-start}
.upload-left{flex:1;min-width:240px}
.upload-right{flex:1.5;display:flex;justify-content:center;align-items:center}
.upload-btn{display:inline-block;padding:10px 24px;background:#667eea;color:#fff;border:none;border-radius:8px;font-size:15px;cursor:pointer}
.upload-btn:hover{opacity:0.9}
#file-input{display:none}
.preview-box{border:2px dashed #ddd;border-radius:10px;min-height:260px;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#fafafa;max-width:400px}
.preview-box img{max-width:100%;max-height:280px;object-fit:contain}
.preview-box .placeholder{color:#bbb;font-size:14px;text-align:center;padding:20px}
.controls{margin-top:16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.controls label{display:flex;align-items:center;gap:6px;font-size:14px;cursor:pointer}
.controls input[type=checkbox]{width:18px;height:18px;cursor:pointer}
.btn-generate{padding:10px 32px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;border-radius:8px;font-size:16px;cursor:pointer;font-weight:600}
.btn-generate:hover{opacity:0.9}
.btn-generate:disabled{opacity:0.5;cursor:not-allowed}
.status{font-size:13px;color:#666;margin-top:8px;padding:8px 12px;border-radius:6px}
.status.loading{background:#fff3cd;color:#856404}
.status.success{background:#d4edda;color:#155724}
.status.error{background:#f8d7da;color:#721c24}

/* 结果区 */
.results-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:12px}
.result-card{background:#fafafa;border-radius:10px;padding:10px;text-align:center;border:1px solid #eee}
.result-card .angle-label{font-weight:600;color:#555;margin-bottom:8px;font-size:14px}
.result-card img{max-width:100%;max-height:280px;border-radius:6px;object-fit:contain;background:#fff}
.result-card .empty{color:#ccc;font-size:13px;padding:60px 0}

/* 进度条 */
.progress-bar{width:100%;height:6px;background:#e9ecef;border-radius:3px;overflow:hidden;margin:12px 0;display:none}
.progress-bar .fill{height:100%;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:3px;transition:width 0.3s;width:0%}

/* 历史记录 */
.history-section{margin-top:24px}
.history-section h2{font-size:18px;margin-bottom:12px;color:#444}
.history-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.history-item{border:1px solid #eee;border-radius:8px;padding:8px;cursor:pointer;transition:box-shadow 0.2s;background:#fff}
.history-item:hover{box-shadow:0 2px 8px rgba(0,0,0,0.12)}
.history-item img{width:100%;height:100px;object-fit:cover;border-radius:4px}
.history-item .h-time{font-size:11px;color:#999;margin-top:4px;text-align:center}
.history-angles{display:flex;gap:2px;margin-top:4px}
.history-angles img{width:33%;height:50px;object-fit:cover;border-radius:3px}

/* 详情弹窗 */
.modal-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);z-index:999;justify-content:center;align-items:center}
.modal-overlay.active{display:flex}
.modal{background:#fff;border-radius:16px;padding:24px;max-width:900px;width:90%;max-height:85vh;overflow-y:auto;position:relative}
.modal-close{position:absolute;top:12px;right:16px;font-size:24px;cursor:pointer;color:#999;background:none;border:none}
.modal-close:hover{color:#333}
.modal-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:16px}
.modal-card{text-align:center}
.modal-card p{font-weight:600;color:#555;margin-bottom:8px}
.modal-card img{max-width:100%;max-height:300px;border-radius:8px}
.modal-info{color:#888;font-size:13px;margin-top:12px;text-align:center}

/* 空状态 */
.empty-state{text-align:center;padding:40px;color:#ccc;font-size:14px}

/* Tab 切换 */
.tabs{display:flex;gap:0;margin-bottom:16px;border-bottom:2px solid #eee}
.tab{padding:8px 20px;cursor:pointer;color:#888;font-size:14px;border-bottom:2px solid transparent;margin-bottom:-2px}
.tab.active{color:#667eea;border-bottom-color:#667eea;font-weight:600}
.tab-content{display:none}
.tab-content.active{display:block}

@media(max-width:768px){
  .upload-area{flex-direction:column}
  .results-grid{grid-template-columns:1fr}
  .history-grid{grid-template-columns:repeat(auto-fill,minmax(120px,1fr))}
  .modal-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="container">
<header>
  <h1>文生图智能体</h1>
  <p>上传面部照片 → 自动生成右转 · 端正 · 左转三个角度</p>
</header>

<div class="card">
  <div class="upload-area">
    <div class="upload-left">
      <label class="upload-btn" onclick="document.getElementById('file-input').click()">选择照片</label>
      <input type="file" id="file-input" accept="image/*" onchange="onFileSelect(event)">
      <div class="controls">
        <label><input type="checkbox" id="outpaint-check" checked> 执行扩图</label>
        <button class="btn-generate" id="gen-btn" onclick="generate()" disabled>开始生成</button>
      </div>
      <div id="status" class="status"></div>
      <div class="progress-bar" id="progress-bar"><div class="fill" id="progress-fill"></div></div>
    </div>
    <div class="upload-right">
      <div class="preview-box" id="preview-box">
        <div class="placeholder">选择照片后在此预览<br>支持 JPG / PNG</div>
      </div>
    </div>
  </div>
</div>

<div id="result-section" style="display:none">
  <div class="card">
    <h2 style="font-size:16px;color:#444;margin-bottom:12px">生成结果</h2>
    <div class="results-grid">
      <div class="result-card"><div class="angle-label">左转</div><img id="result-left" class="empty" src=""><div class="empty">等待生成...</div></div>
      <div class="result-card"><div class="angle-label">端正</div><img id="result-front" class="empty" src=""><div class="empty">等待生成...</div></div>
      <div class="result-card"><div class="angle-label">右转</div><img id="result-right" class="empty" src=""><div class="empty">等待生成...</div></div>
    </div>
  </div>
  <div class="card" id="outpaint-section" style="display:none">
    <h2 style="font-size:16px;color:#444;margin-bottom:12px">扩图结果</h2>
    <img id="result-outpaint" style="max-width:100%;border-radius:8px">
  </div>
</div>

<div class="card history-section">
  <h2>历史记录</h2>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('recent')">最近生成</div>
    <div class="tab" onclick="switchTab('all')">全部记录</div>
  </div>
  <div id="tab-recent" class="tab-content active"></div>
  <div id="tab-all" class="tab-content"></div>
</div>
</div>

<!-- 详情弹窗 -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <div id="modal-content"></div>
  </div>
</div>

<script>
let currentImageData = null;
let currentFile = null;

function onFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  currentFile = file;
  document.getElementById('gen-btn').disabled = false;

  const reader = new FileReader();
  reader.onload = function(ev) {
    currentImageData = ev.target.result;
    document.getElementById('preview-box').innerHTML = '<img src="' + ev.target.result + '">';
  };
  reader.readAsDataURL(file);
  setStatus('已选择: ' + file.name, '');
}

function setStatus(msg, type) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status' + (type ? ' ' + type : '');
}

function setProgress(pct) {
  const bar = document.getElementById('progress-bar');
  const fill = document.getElementById('progress-fill');
  bar.style.display = 'block';
  fill.style.width = pct + '%';
}

async function generate() {
  if (!currentFile) return;
  document.getElementById('gen-btn').disabled = true;
  document.getElementById('result-section').style.display = 'none';

  const fd = new FormData();
  fd.append('image', currentFile);
  fd.append('outpaint', document.getElementById('outpaint-check').checked ? '1' : '0');

  setStatus('正在调用AI生成（约2-3分钟），请稍候...', 'loading');
  setProgress(10);

  try {
    const resp = await fetch('/generate', { method: 'POST', body: fd });
    setProgress(90);
    const data = await resp.json();
    setProgress(100);

    if (data.error) {
      setStatus('错误: ' + data.error, 'error');
      document.getElementById('gen-btn').disabled = false;
      return;
    }

    // 显示结果
    if (data.left && data.left.startsWith('data:image')) {
      document.getElementById('result-left').src = data.left;
      document.getElementById('result-left').className = '';
    } else {
      document.getElementById('result-left').className = 'empty';
      document.getElementById('result-left').src = '';
    }

    if (data.front && data.front.startsWith('data:image')) {
      document.getElementById('result-front').src = data.front;
      document.getElementById('result-front').className = '';
    } else {
      document.getElementById('result-front').className = 'empty';
      document.getElementById('result-front').src = '';
    }

    if (data.right && data.right.startsWith('data:image')) {
      document.getElementById('result-right').src = data.right;
      document.getElementById('result-right').className = '';
    } else {
      document.getElementById('result-right').className = 'empty';
      document.getElementById('result-right').src = '';
    }

    document.getElementById('result-section').style.display = 'block';

    if (data.outpaint) {
      document.getElementById('result-outpaint').src = data.outpaint;
      document.getElementById('outpaint-section').style.display = 'block';
    }

    setStatus('生成完成！', 'success');

    // 刷新历史
    loadHistory();

    // 延迟隐藏进度条
    setTimeout(() => { setProgress(0); document.getElementById('progress-bar').style.display = 'none'; }, 1000);
  } catch (err) {
    setStatus('请求失败: ' + err.message, 'error');
  }
  document.getElementById('gen-btn').disabled = false;
}

async function loadHistory() {
  try {
    const resp = await fetch('/history');
    const data = await resp.json();
    renderHistory(data);
  } catch(e) {}
}

function renderHistory(records) {
  const recentEl = document.getElementById('tab-recent');
  const allEl = document.getElementById('tab-all');

  if (!records || records.length === 0) {
    recentEl.innerHTML = '<div class="empty-state">暂无历史记录，生成照片后将自动保存</div>';
    allEl.innerHTML = '<div class="empty-state">暂无历史记录</div>';
    return;
  }

  // 最近5条
  let recentHtml = '<div class="history-grid">';
  for (let i = 0; i < Math.min(records.length, 5); i++) {
    recentHtml += buildHistoryItem(records[i], i);
  }
  recentHtml += '</div>';
  recentEl.innerHTML = recentHtml;

  // 全部
  let allHtml = '<div class="history-grid">';
  for (let i = 0; i < records.length; i++) {
    allHtml += buildHistoryItem(records[i], i);
  }
  allHtml += '</div>';
  allEl.innerHTML = allHtml;

  // 绑定点击事件
  document.querySelectorAll('.history-item').forEach(el => {
    el.addEventListener('click', function() {
      const idx = this.dataset.index;
      showHistoryDetail(records[parseInt(idx)]);
    });
  });
}

function buildHistoryItem(rec, idx) {
  const time = rec.time || '';
  const frontImg = rec.images && rec.images.front ? rec.images.front : '';
  return '<div class="history-item" data-index="' + idx + '">' +
    (frontImg ? '<img src="' + frontImg + '">' : '<div style="height:100px;background:#f5f5f5;border-radius:4px"></div>') +
    '<div class="h-time">' + time + '</div>' +
    '</div>';
}

function showHistoryDetail(rec) {
  const el = document.getElementById('modal-content');
  const imgs = rec.images || {};
  const hasOutpaint = rec.outpaint_images && rec.outpaint_images.front;
  let html = '<h2 style="text-align:center;margin-bottom:4px">历史记录详情</h2>';
  html += '<p class="modal-info">生成时间: ' + (rec.time || '') + '</p>';

  html += '<div class="modal-grid">';
  html += '<div class="modal-card"><p>左转</p><img src="' + (imgs.left || '') + '"></div>';
  html += '<div class="modal-card"><p>端正</p><img src="' + (imgs.front || '') + '"></div>';
  html += '<div class="modal-card"><p>右转</p><img src="' + (imgs.right || '') + '"></div>';
  html += '</div>';

  if (hasOutpaint) {
    html += '<h3 style="margin-top:20px;color:#555;font-size:15px">扩图结果</h3>';
    html += '<div class="modal-grid">';
    html += '<div class="modal-card"><p>左转扩图</p><img src="' + (rec.outpaint_images.left || '') + '"></div>';
    html += '<div class="modal-card"><p>端正扩图</p><img src="' + (rec.outpaint_images.front || '') + '"></div>';
    html += '<div class="modal-card"><p>右转扩图</p><img src="' + (rec.outpaint_images.right || '') + '"></div>';
    html += '</div>';
  }

  el.innerHTML = html;
  document.getElementById('modal').classList.add('active');
}

function closeModal() {
  document.getElementById('modal').classList.remove('active');
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('.tab[onclick*="' + name + '"]').classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

// 点击弹窗背景关闭
document.getElementById('modal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});

// 初始化
loadHistory();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))

        elif path == "/history":
            history = load_history()
            self.send_json(history)

        elif path.startswith("/images/"):
            # 提供历史图片访问
            safe_path = os.path.normpath(os.path.join(BASE_DIR, path.lstrip("/")))
            if safe_path.startswith(BASE_DIR) and os.path.exists(safe_path):
                self.send_response(200)
                if path.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif path.endswith(".jpg") or path.endswith(".jpeg"):
                    self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(safe_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif path == "/test":
            self._send_text("服务器运行正常！")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/generate":
            self._handle_generate()
        elif self.path == "/generate_json":
            self._handle_generate_json()
        else:
            self.send_error(404)

    def _handle_generate(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            boundary = self.headers.get("Content-Type", "").split("boundary=")[-1]

            image_data = None
            do_outpaint = True
            preview = False  # 是否返回缩略图（小尺寸）

            for part in body.split(b"--" + boundary.encode()):
                if b'name="image"' in part:
                    idx = part.find(b"\r\n\r\n") + 4
                    if idx > 4:
                        raw = part[idx:].rstrip(b"\r\n- \t")
                        image_data = raw
                if b'name="outpaint"' in part:
                    idx = part.find(b"\r\n\r\n") + 4
                    if idx > 4:
                        val = part[idx:].strip().decode()
                        do_outpaint = (val == "1")
                if b'name="preview"' in part:
                    idx = part.find(b"\r\n\r\n") + 4
                    if idx > 4:
                        val = part[idx:].strip().decode()
                        preview = (val == "1")

            if not image_data:
                self.send_json({"error": "未找到图片"})
                return

            # 检查输出目录
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"[DEBUG] 输出目录: {OUTPUT_DIR}")

            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            agent = get_agent()
            result = agent.run(None, do_outpainting=do_outpaint, image_obj=img)

            # 调试日志
            print(f"[DEBUG] result keys: {list(result.keys())}")
            print(f"[DEBUG] rotated keys: {list(result['rotated'].keys())}")

            # 编码结果图片为 base64 data URI
            def img_to_b64(path, is_preview=False):
                try:
                    if not os.path.exists(path):
                        print(f"[ERROR] 文件不存在: {path}")
                        return ""
                    if is_preview:
                        # 缩略图：最长边 300px，大幅减小 base64 体积
                        thumb = Image.open(path)
                        thumb.thumbnail((300, 300))
                        buf = io.BytesIO()
                        thumb.save(buf, format="PNG")
                        data = buf.getvalue()
                        return "data:image/png;base64," + base64.b64encode(data).decode()
                    with open(path, "rb") as f:
                        data = f.read()
                        return "data:image/png;base64," + base64.b64encode(data).decode()
                except Exception as e:
                    print(f"[ERROR] 读取图片失败 {path}: {e}")
                    return ""

            right_b64 = img_to_b64(result["rotated"]["right"], preview)
            front_b64 = img_to_b64(result["rotated"]["front"], preview)
            left_b64 = img_to_b64(result["rotated"]["left"], preview)

            print(
                f"[DEBUG] right length: {len(right_b64)}, front length: {len(front_b64)}, left length: {len(left_b64)}")

            response = {
                "right": right_b64,
                "front": front_b64,
                "left": left_b64,
            }

            # 扩图结果
            outpaint_data = {}
            if do_outpaint and result.get("outpainted"):
                for angle in ["right", "front", "left"]:
                    if angle in result["outpainted"]:
                        outpaint_data[angle] = img_to_b64(result["outpainted"][angle], preview)
                # 拼接成一张全景图
                if len(outpaint_data) == 3:
                    try:
                        w, h = Image.open(result["outpainted"]["right"]).size
                        canvas = Image.new("RGB", (w * 3, h))
                        canvas.paste(Image.open(result["outpainted"]["right"]), (0, 0))
                        canvas.paste(Image.open(result["outpainted"]["front"]), (w, 0))
                        canvas.paste(Image.open(result["outpainted"]["left"]), (w * 2, 0))
                        buf = io.BytesIO()
                        canvas.save(buf, format="PNG")
                        response["outpaint"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                    except Exception as e:
                        print(f"[ERROR] 拼接扩图失败: {e}")

            # 保存历史记录
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "time": now,
                "images": {
                    "right": right_b64,
                    "front": front_b64,
                    "left": left_b64,
                },
                "outpaint_images": outpaint_data if outpaint_data else {},
                "do_outpainting": do_outpaint,
            }
            save_history(record)

            self.send_json(response)

        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e)})

    def _handle_generate_json(self):
        """JSON base64 接口：供 Dify Workflow Code 节点调用"""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))

            image_b64 = body.get("image_b64", "") or body.get("image", "")
            do_outpaint = body.get("do_outpaint", True)

            if not image_b64:
                self.send_json({"success": False, "error": "缺少 image_b64 参数"})
                return

            # 解码 base64 图片
            import base64 as b64mod
            raw = b64mod.b64decode(image_b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")

            # 调用 Agent
            agent = get_agent()
            result = agent.run(None, do_outpainting=bool(do_outpaint), image_obj=img)

            def img_to_b64(path):
                try:
                    if not os.path.exists(path):
                        return ""
                    with open(path, "rb") as f:
                        data = f.read()
                        return "data:image/png;base64," + b64mod.b64encode(data).decode()
                except Exception:
                    return ""

            response = {
                "success": True,
                "images": {
                    "right": img_to_b64(result["rotated"]["right"]),
                    "front": img_to_b64(result["rotated"]["front"]),
                    "left": img_to_b64(result["rotated"]["left"]),
                },
                "outpainted": {},
            }

            if do_outpaint and result.get("outpainted"):
                for angle in ["right", "front", "left"]:
                    if angle in result["outpainted"]:
                        response["outpainted"][angle] = img_to_b64(result["outpainted"][angle])

            self.send_json(response)

        except json.JSONDecodeError:
            self.send_json({"success": False, "error": "请求体必须是 JSON"})
        except Exception as e:
            traceback.print_exc()
            self.send_json({"success": False, "error": str(e)})

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _send_text(self, text):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def log_message(self, fmt, *args):
        try:
            safe = [str(a) if a is not None else "" for a in args]
            while len(safe) < 3:
                safe.append("")
            print(f"[WebUI] {safe[0]} {safe[1]} {safe[2]}")
        except:
            print(f"[WebUI] {fmt}")


if __name__ == "__main__":
    port = 7860
    server = HTTPServer(("0.0.0.0", port), Handler)
    print("=" * 55)
    print("  文生图智能体 - API服务")
    print(f"  API地址: http://0.0.0.0:{port}")
    print("=" * 55)
    print("  (按 Ctrl+C 停止)")
    print("=" * 55)
    server.serve_forever()
