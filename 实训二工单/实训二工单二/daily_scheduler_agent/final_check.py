"""Final verification from Windows CMD"""
import subprocess

project = r'C:\Users\freedom\Desktop\实训二工单二\daily_scheduler_agent'

# Check from cmd.exe
checks = [
    ('start_web.bat', f'type "{project}\\start_web.bat"'),
    ('notification.py', f'dir "{project}\\reminder\\notification.py"'),
    ('index.html: notifyBtn', f'findstr /N "notifyBtn" "{project}\\templates\\index.html"'),
    ('index.html: micBtn', f'findstr /N "micBtn" "{project}\\templates\\index.html"'),
    ('web_app: test_notification', f'findstr /N "test_notification" "{project}\\web_app.py"'),
]

for name, cmd in checks:
    r = subprocess.run(['cmd.exe', '/c', cmd], capture_output=True, text=True, timeout=5, cwd='C:\\')
    out = r.stdout.strip()[:200]
    err = r.stderr.strip()[:200]
    status = 'OK' if r.returncode == 0 else 'FAIL'
    print(f'[{status}] {name}')
    if out:
        print(f'  -> {out}')
    if err:
        print(f'  !! {err}')

# Test notification module import
print('\n--- Testing notification module ---')
r2 = subprocess.run(
    ['cmd.exe', '/c', 
     f'C:\\Users\\freedom\\.conda\\envs\\py310\\python.exe -c "import sys; sys.path.insert(0,\'{project}\'); from reminder.notification import show_windows_toast, notify_reminder; print(\'Module OK\')"'],
    capture_output=True, text=True, timeout=10, cwd='C:\\')
print(r2.stdout.strip()[:200])
if r2.stderr.strip():
    print('STDERR:', r2.stderr.strip()[:200])

print('\n=== Verification complete ===')
