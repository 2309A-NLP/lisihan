# -*- coding: utf-8 -*-
"""从 Windows Python 直接写入所有改动"""
import os
import sys

PROJECT = r'C:\Users\freedom\Desktop\实训二工单二\daily_scheduler_agent'

sys.stdout.reconfigure(encoding='utf-8')

# 1. start_web.bat
bat = '@echo off\r\n'
bat += 'cd /d C:\\Users\\freedom\\Desktop\\实训二工单二\\daily_scheduler_agent\r\n'
bat += '\r\n'
bat += 'C:\\Users\\freedom\\.conda\\envs\\py310\\python.exe web_app.py\r\n'
bat += '\r\n'
bat += 'pause\r\n'
with open(os.path.join(PROJECT, 'start_web.bat'), 'w', newline='\r\n') as f:
    f.write(bat)
print('[OK] start_web.bat')

# 2. reminder/notification.py
notif = '''# -*- coding: utf-8 -*-
import logging
import subprocess
import threading

logger = logging.getLogger("Notification")
_POWERSHELL = r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"

def show_windows_toast(title, message, duration=8):
    def _show():
        try:
            safe_title = title.replace('"', '`"')
            safe_message = message.replace('"', '`"')
            ps = (
                'Add-Type -AssemblyName System.Windows.Forms\\n'
                '$notify = New-Object System.Windows.Forms.NotifyIcon\\n'
                '$notify.Icon = [System.Drawing.SystemIcons]::Information\\n'
                '$notify.BalloonTipTitle = "' + safe_title + '"\\n'
                '$notify.BalloonTipText = "' + safe_message + '"\\n'
                '$notify.Visible = $true\\n'
                '$notify.ShowBalloonTip(' + str(duration * 1000) + ')\\n'
                'Start-Sleep -Seconds ' + str(duration) + '\\n'
                '$notify.Dispose()\\n'
            )
            subprocess.run(
                [_POWERSHELL, "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                capture_output=True, timeout=duration + 5,
            )
            logger.info("Windows toast sent: %s - %s", title, message)
        except subprocess.TimeoutExpired:
            logger.warning("Windows toast timed out (duration=%ss)", duration)
        except Exception as exc:
            logger.warning("Windows toast failed: %s", exc)
    threading.Thread(target=_show, daemon=True).start()

def notify_reminder(schedule_id, content, scheduled_time):
    show_windows_toast("日程提醒", content + "\\n时间：" + scheduled_time)
'''
with open(os.path.join(PROJECT, 'reminder', 'notification.py'), 'w', encoding='utf-8') as f:
    f.write(notif)
print('[OK] reminder/notification.py')

# 3. web_app.py
wp = os.path.join(PROJECT, 'web_app.py')
with open(wp, 'r', encoding='utf-8') as f:
    code = f.read()

if 'from reminder.notification import' not in code:
    code = code.replace(
        'from reminder.message_templates import build_reminder_message',
        'from reminder.message_templates import build_reminder_message\\nfrom reminder.notification import notify_reminder'
    )

if 'test_notification' not in code:
    route = '''

@app.route("/test_notification", methods=["GET"])
def test_notification():
    from reminder.notification import show_windows_toast
    show_windows_toast("日程提醒", "测试通知 - 弹窗功能正常！", duration=5)
    return jsonify({"ok": True, "reply": "测试通知已发送，请查看电脑右下角系统托盘区域。"})
'''
    code = code.replace(
        'return jsonify(get_database_status())\\n\\n\\n@app.route("/logs"',
        'return jsonify(get_database_status())' + route + '\\n\\n@app.route("/logs"'
    )

call = '''                # 发送 Windows 右下角 Toast 弹窗通知
                notify_reminder(
                    schedule_id=schedule["id"],
                    content=schedule["content"],
                    scheduled_time=format_schedule_time(schedule["occurrence_time"]),
                )'''

old_block = '''                reminders_data.append(
                    {
                        "schedule_id": schedule["id"],
                        "message": message,
                        "scheduled_time": format_schedule_time(schedule["occurrence_time"]),
                    }
                )
                logger.info("Web reminder sent: schedule_id=%s message=%s", schedule["id"], message)'''

new_block = '''                reminders_data.append(
                    {
                        "schedule_id": schedule["id"],
                        "message": message,
                        "scheduled_time": format_schedule_time(schedule["occurrence_time"]),
                    }
                )
                # 发送 Windows 右下角 Toast 弹窗通知
                notify_reminder(
                    schedule_id=schedule["id"],
                    content=schedule["content"],
                    scheduled_time=format_schedule_time(schedule["occurrence_time"]),
                )
                logger.info("Web reminder sent: schedule_id=%s message=%s", schedule["id"], message)'''

code = code.replace(old_block, new_block)

with open(wp, 'w', encoding='utf-8') as f:
    f.write(code)
print('[OK] web_app.py')

# 4. templates/index.html
html_path = os.path.join(PROJECT, 'templates', 'index.html')
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# CSS for mic-btn
if '.mic-btn' not in html:
    html = html.replace(
        'font-weight: 700;\\n        }',
        'font-weight: 700;\\n        }\\n\\n'
        '.mic-btn { width: 50px; min-width: 50px; height: 50px; border: 0; '
        'border-radius: 50%; font-size: 22px; cursor: pointer; background: #f1f5f9; '
        'color: #475569; transition: .18s; display: flex; align-items: center; '
        'justify-content: center; box-shadow: 0 2px 6px rgba(0,0,0,.08); }\\n'
        '.mic-btn:hover { background: #e2e8f0; transform: scale(1.05); }\\n'
        '.mic-btn.listening { background: #ef4444; color: #fff; '
        'box-shadow: 0 0 0 4px rgba(239,68,68,.25); animation: micPulse 1.2s infinite; }\\n'
        '@keyframes micPulse { 0%,100% { box-shadow: 0 0 0 4px rgba(239,68,68,.25); } '
        '50% { box-shadow: 0 0 0 8px rgba(239,68,68,.12); } }\\n'
        '.mic-btn:disabled { opacity: .5; cursor: not-allowed; animation: none; }',
        1
    )

# notify button in header
if 'notifyBtn' not in html:
    html = html.replace(
        '<button class="header-btn" id="statusBtn" type="button">系统状态</button>',
        '<button class="header-btn" id="statusBtn" type="button">系统状态</button>\\n'
        '<button class="header-btn" id="notifyBtn" type="button">测试弹窗</button>'
    )

# mic button in composer
if 'micBtn' not in html:
    html = html.replace(
        '<button class="send-btn" id="sendBtn" type="submit">发送</button>',
        '<button class="mic-btn" id="micBtn" type="button" title="语音输入">[mic]</button>\\n'
        '<button class="send-btn" id="sendBtn" type="submit">发送</button>'
    )

# JS reference
if 'const notifyBtn' not in html:
    html = html.replace(
        'const statusBtn = document.getElementById("statusBtn");',
        'const statusBtn = document.getElementById("statusBtn");\\n'
        'const notifyBtn = document.getElementById("notifyBtn");\\n'
        'const micBtn = document.getElementById("micBtn");'
    )

# notifyBtn event handler
if 'notifyBtn.addEventListener' not in html:
    njs = '''
notifyBtn.addEventListener("click", async () => {
    setBusy(notifyBtn, "发送中", true);
    try {
        const data = await getJson("/test_notification");
        appendMessage("assistant", data.reply);
    } catch (error) {
        appendMessage("assistant", error.message);
    } finally {
        setBusy(notifyBtn, "发送中", false);
        input.focus();
    }
});'''
    html = html.replace(
        'window.addEventListener("load"',
        njs + '\\n\\nwindow.addEventListener("load"'
    )

# Voice input JS
if '---- 语音输入 ----' not in html:
    vjs = '''
// ---- 语音输入 ----
let recognition = null;
let isListening = false;
function initSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { micBtn.style.display = "none"; return null; }
    const r = new SR();
    r.lang = "zh-CN"; r.continuous = false; r.interimResults = false; r.maxAlternatives = 1;
    return r;
}
micBtn.addEventListener("click", () => {
    if (isListening) { try { recognition.stop(); } catch(e){} micBtn.classList.remove("listening"); isListening = false; micBtn.title = "语音输入"; return; }
    if (!recognition) {
        recognition = initSpeechRecognition();
        if (!recognition) return;
        recognition.onresult = function(e) {
            var text = e.results[0][0].transcript;
            input.value = "";
            micBtn.classList.remove("listening");
            isListening = false;
            micBtn.title = "语音输入";
            sendMessage(text);
        };
        recognition.onerror = function(e) {
            micBtn.classList.remove("listening");
            isListening = false;
            micBtn.title = "语音输入";
        };
        recognition.onend = function() {
            micBtn.classList.remove("listening");
            isListening = false;
            micBtn.title = "语音输入";
        };
    }
    try { recognition.start(); isListening = true; micBtn.classList.add("listening"); micBtn.title = "点击停止录音"; } catch(e){}
});'''

    html = html.replace(
        'form.addEventListener("submit"',
        vjs + '\\n\\nform.addEventListener("submit"'
    )

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print('[OK] templates/index.html')

# Verification
print('\\n=== Verification ===')
for name, path in [
    ('start_web.bat', os.path.join(PROJECT, 'start_web.bat')),
    ('notification.py', os.path.join(PROJECT, 'reminder', 'notification.py')),
    ('web_app.py', os.path.join(PROJECT, 'web_app.py')),
    ('index.html', os.path.join(PROJECT, 'templates', 'index.html')),
]:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f'  {name}: exists={exists}, size={size}')

# Check content markers
with open(os.path.join(PROJECT, 'web_app.py'), 'r', encoding='utf-8') as f:
    wc = f.read()
print(f'  web_app has test_notification: {"test_notification" in wc}')

with open(os.path.join(PROJECT, 'templates', 'index.html'), 'r', encoding='utf-8') as f:
    hc = f.read()
print(f'  index has notifyBtn: {"notifyBtn" in hc}')
print(f'  index has micBtn: {"micBtn" in hc}')

print('\\n[DONE] All changes applied from Windows side.')
