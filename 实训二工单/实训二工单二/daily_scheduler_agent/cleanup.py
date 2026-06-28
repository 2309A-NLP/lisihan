"""Kill old Flask processes"""
import subprocess

r = subprocess.run(['cmd.exe', '/c', 'tasklist /FI "IMAGENAME eq python.exe" /FO CSV'],
                   capture_output=True, text=True, timeout=5, cwd='C:\\')
print('Running Python processes:')
for line in r.stdout.splitlines():
    if 'python' in line.lower() and ',' in line:
        parts = line.split(',')
        pid = parts[1].strip().strip('"')
        name = parts[0].strip().strip('"')
        print(f'  {name} (PID: {pid})')

# Kill any that might be web_app
r2 = subprocess.run(['cmd.exe', '/c', 'taskkill /F /FI "IMAGENAME eq python.exe" 2>nul || echo none'],
                    capture_output=True, text=True, timeout=5, cwd='C:\\')
print('\nKill result:', r2.stdout.strip()[:200])
