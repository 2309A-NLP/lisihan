"""Verify files from Windows side"""
import subprocess
import os

project = r'C:\Users\freedom\Desktop\实训二工单二\daily_scheduler_agent'

# 1. Clear all __pycache__  
print("=== Clearing pycache ===")
for root, dirs, files in os.walk(project):
    for d in dirs:
        if d == '__pycache__':
            path = os.path.join(root, d)
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
            os.rmdir(path)
            print(f'  Removed {path}')

# 2. Verify index.html from cmd.exe
print("\n=== index.html notifyBtn check ===")
r = subprocess.run(
    ['cmd.exe', '/c', f'findstr /N "notifyBtn" "{project}\\templates\\index.html"'],
    capture_output=True, text=True, timeout=5, cwd='C:\\'
)
print(r.stdout[:500] if r.stdout else '(no matches)')

# 3. Verify notification.py
print("\n=== notification.py exists check ===")
r2 = subprocess.run(
    ['cmd.exe', '/c', f'if exist "{project}\\reminder\\notification.py" (echo EXISTS) else (echo NOT FOUND)'],
    capture_output=True, text=True, timeout=5, cwd='C:\\'
)
print(r2.stdout.strip())

# 4. Check start_web.bat content
print("\n=== start_web.bat ===")
r3 = subprocess.run(
    ['cmd.exe', '/c', f'type "{project}\\start_web.bat"'],
    capture_output=True, text=True, timeout=5, cwd='C:\\'
)
print(r3.stdout.strip())

# 5. Check web_app.py has new endpoint
print("\n=== web_app.py test_notification ===")
r4 = subprocess.run(
    ['cmd.exe', '/c', f'findstr /N "test_notification" "{project}\\web_app.py"'],
    capture_output=True, text=True, timeout=5, cwd='C:\\'
)
print(r4.stdout[:300] if r4.stdout else '(no matches)')

print("\n=== Verification complete ===")
