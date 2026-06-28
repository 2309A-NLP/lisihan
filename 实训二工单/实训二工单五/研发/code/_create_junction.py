"""Create a junction for the project"""
import subprocess, os

# Change to C: drive first to avoid UNC path issue
cmd = 'C: && cd \\ && if exist C:\\Users\\freedom\\Desktop\\zhaogu rmdir C:\\Users\\freedom\\Desktop\\zhaogu && mklink /J C:\\Users\\freedom\\Desktop\\zhaogu "C:\\Users\\freedom\\Desktop\\招股书问答智能体"'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='C:\\')
print('stdout:', result.stdout)
print('stderr:', result.stderr[:200] if result.stderr else '')
print('return:', result.returncode)

# Verify
p = r'C:\Users\freedom\Desktop\zhaogu\output\prospectus_index.faiss'
print(f'\nExists: {os.path.exists(p)}')
if os.path.exists(p):
    import faiss
    i = faiss.read_index(p)
    print(f'FAISS OK: {i.ntotal}')
