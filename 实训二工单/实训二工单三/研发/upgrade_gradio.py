import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", "gradio"],
    capture_output=True, text=True
)
print(result.stdout[-200:] if result.stdout else "")
print(result.stderr[-200:] if result.stderr else "")
print(f"return code: {result.returncode}")
