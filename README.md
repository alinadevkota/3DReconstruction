# MatProp3D: Learning Material Properties for interactive 3D reconstruction

# Clone Repository
```bash
git clone --recursive https://github.com/alinadevkota/3DReconstruction.git
git submodule update --init --recursive
```

# Environment Setup
```bash
conda create -n matprop3d python=3.1
conda activate matprop3d
pip install --upgrade pip setuptools
pip install -e .
```

Make an empty clean directory to use as project workspace where you store all the outputs.
```bash
export MATPROP3DWS=/path/to/workspace/dir
```

# Test Scene Renders

Test if the Warp and its dependencies is working well.
```bash
python scripts/render_playground.py
```
You should see simulation similar to this:

![](media/warp_multi_ball.gif)

# Learning Ball Density (in Simulation)

```bash
python3 scripts/learn_density_sim.py
```
You should see simulation similar to this:

![](media/two_ball_pool.gif)

After the training iterations completed, you will see following graphs and estimated ball densities.

Here Ground Truth densities of two balls: 185, 200

Initial Guess: 500, 5000

Estimated values: 186.25, 200.28

![](media/density_estimation.png)

# Real Data Preparation

First, collect bunch of images of an static scene. For this you can capture images of a target object from multiple views and put it inside a directory. Or you can capture a video an then extract the video frames using script below.
```
python scripts/video_frame_extractor.py --video-file /path/to/video/file --output-dir /path/to/ws/outputdir
```

# Static 3D Reconstruction using Gaussian Splatting

From a set of images, you can do point cloud reconstruction using COLMAP which also recovers the camera poses that can be used to train Gaussian Splatting.

Run COLMAP:

```bash
ns-process-data images --data $WORKSPACE/images --output-dir $WORKSPACE/colmap
```

Train a Gaussain Splatting Model:

```bash
ns-train splatfacto --data $WORKSPACE/colmap --output-dir $WORKSPACE/gaussian_splats
```

Save the ply file at: `$WORKSPACE/gaussian.ply`

# Learning Ball Density (in Real)

```bash
python scripts/track_ball.py --video-file data/ball_drop.mp4
```
![](media/real_ball_traj.png)

```bash
python3 scripts/learn_density_real.py
```

![](media/real_ball_drop_in_sim.gif)

This gives density estimation plot as such:

![](media/density_estimation_real.png)


# Integrate Material Property with Static Gaussian Splats

Follow the tutorial [here](https://github.com/rashikshrestha/Interact3D) to inject density to get dynamic gaussian splat renders.

![](media/tennis_drop.gif)

