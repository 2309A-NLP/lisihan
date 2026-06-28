try:
    import gradio
    print(f"gradio {gradio.__version__} 已安装")
except ImportError:
    print("gradio 未安装")
