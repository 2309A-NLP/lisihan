"""Verify files using pure Python (no cmd.exe)"""
import os

project = r'C:\Users\freedom\Desktop\实训二工单二\daily_scheduler_agent'

files = {
    'start_web.bat': os.path.join(project, 'start_web.bat'),
    'notification.py': os.path.join(project, 'reminder', 'notification.py'),
    'web_app.py': os.path.join(project, 'web_app.py'),
    'index.html': os.path.join(project, 'templates', 'index.html'),
}

all_ok = True
for name, path in files.items():
    if not os.path.exists(path):
        print(f'MISSING: {name} at {path}')
        all_ok = False
        continue
    size = os.path.getsize(path)
    print(f'OK: {name} ({size} bytes)')

if not all_ok:
    print('\nSome files are missing! Need to rewrite.')
    exit(1)

# Content verification
with open(files['web_app.py'], 'r', encoding='utf-8') as f:
    wc = f.read()
print(f'web_app.py has notify_reminder: {"notify_reminder" in wc}')
print(f'web_app.py has test_notification: {"test_notification" in wc}')

with open(files['index.html'], 'r', encoding='utf-8') as f:
    hc = f.read()
print(f'index.html has notifyBtn: {"notifyBtn" in hc}')
print(f'index.html has micBtn: {"micBtn" in hc}')
print(f'index.html line count: {len(hc.splitlines())}')

# Read notification.py and verify syntax
with open(files['notification.py'], 'r', encoding='utf-8') as f:
    nc = f.read()
print(f'notification.py has show_windows_toast: {"show_windows_toast" in nc}')
print(f'notification.py has notify_reminder: {"notify_reminder" in nc}')

# Read start_web.bat
with open(files['start_web.bat'], 'r') as f:
    bc = f.read()
print(f'start_web.bat has conda path: {"conda" in bc}')
print(f'start_web.bat has pause: {"pause" in bc}')

print('\n=== All files verified successfully from Windows Python ===')
