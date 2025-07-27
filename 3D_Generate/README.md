## Installations

In our pipeline, we use cotracker2 and you can follow this [instruction](https://github.com/facebookresearch/co-tracker/tree/8d364031971f6b3efec945dd15c468a183e58212) to install.

**Step 1 — create your own environment**
```bash
conda create -n [ENV_NAME] python=3.10
conda activate [ENV_NAME]
```

**Step 2 — install pytorch and related modules**
In this part, we use `Pytorch==2.3.1`, accompanied with `cuda==12.1` and `torchvision==0.18.1`. Please follow the [Pytorch official webste](https://pytorch.org/get-started/previous-versions/) to install certain version.
**Step 3 — install other modules**

```bash
pip install trimesh==4.5.3
pip install imageio==2.36.0
pip install matplotlib==3.9.2
pip install scipy==1.12.0 
pip install numpy==1.26.4
```

**Step 4 - download the dataset**



## Execution

### Tracking Data Generation

Processes video files using CoTracker to generate motion tracking visualizations with interpolated tracks.
```bash
python interpolate_tracks.py \
    --video-path ./input/video.mp4 \
    --save-dir ./output/tracking_results \
    --model-path ./models/cotracker2.pth
```

**Arguments**

| Argument | Description | 
|----------|-------------|
| `--video-path` | Path to input video file | 
| `--save-dir` | Directory to save results | 
| `--model-path` | Path to CoTracker model |

**Output**

`cotracker_output.pt`: Contains tracking data including:

- `pred_tracks`: Predicted tracks
- `pred_visibility`: Visibility information
- `grid_pts`: Grid points
- `interpolated_tracks`: Interpolated tracks

### 3D Displacement Heatmap Generation

Processes 3D mesh sequences and visualizes displacement data.

```bash
python plot_3Dheatmap_ply.py \
    --ply-folder ./input/ply_sequence \
    --deform-pt ./data/deform.pt \
    --x-pt ./data/x.pt \
    --output-dir ./output/displacement_vis
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `--ply-folder` | Folder containing PLY sequence |
| `--deform-pt` | Path to deformation tensor (.pt) | 
| `--x-pt` | Path to x tensor (.pt) | 
| `--output-dir` | Output directory |

**Output**

For each frame in the sequence:

- `frame_XXX_color.ply`: Colored PLY mesh with displacement visualization
- `frame_XXX_hist.png`: Histogram of displacement distribution

### Visualize the 3D Displacement Heatmap

We use the method provided by EndoNeRF team, please refer to their [work](https://github.com/med-air/EndoNeRF), and use vis_pc.py.
