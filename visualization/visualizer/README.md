# 360LayoutVisualizer

This repo is a visualization tool for 360 Manhattan layout based on PyQt5 and OpenGL. The layout format follows <a href='https://github.com/fuenwang/LayoutMP3D'>LayoutMP3D</a>. 
<p align='center'><image src='src/3Dlayout.png' width='100%'></image></p>

Install the locked project environment from the repository root. The Python dependencies for the visualizer are included in the default environment.

```bash
uv sync --frozen
```

Then, from the repository root, run the visualizer for the provided example. A graphical desktop session and the corresponding system OpenGL/X11 libraries are required.

```bash
uv run python visualization/visualizer/visualizer.py \
  --img visualization/visualizer/src/example.jpg \
  --json visualization/visualizer/src/example.json
```
You can use mouse and keyboard to control the camera.
```yaml
w, a, s, d: translate the camera
left-click: rotate the camera
scroll: zoom in/out
```
<p align='center'><image src='src/demo.png' width='50%'></image></p>
