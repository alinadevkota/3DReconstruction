# MatProp3D: Learning Material Properties for interactive 3D reconstruction

# Environment Setup
```bash
conda create -n matprop3d python=3.1
conda activate matprop3d
pip install --upgrade pip setuptools
pip install -e .
```

# Test Scene Renders

Test if the Warp and its dependencies is working well.
```bash
python scripts/render_playground.py
```
You should see simulation similar to this:

![](media/warp_multi_ball.gif)

# Simulation only density learning


