# Volumetric TSDF Fusion in PyTorch
### Fuse RGB-D images using Truncated Sign Distance Function (TSDF) Fusion with Live visualization

This repository is part of *from scratch implementation* initiative and meant for learning and understanding the basics of Truncated Signed Distance Function (TSDF) Fusion. It's concise and lightweight implementation of the algorithm in PyTorch that also showcases some best practices for object-oriented programming.

**Note:** The repository is not intended for production use, as vanilla PyTorch is not the best choice for performance-critical applications. For applications requiring high performance, consider using a CUDA-based implementations.

<br>
<p align="center">
  <img src="output/tsdf_fusion.gif" height=250px/>
</p>

## Installation
1. (Optional) Create new conda environment:
```shell
conda create --name tsdf_fusion python=3.10
conda activate tsdf_fusion
```
2. Install dependencies:
  ```shell
  pip install -r requirements.txt
  ```

## Usage

Command:
```shell
python demo.py [--data_dir DATA_DIR] --visualize {True,False} [--output_dir OUTPUT_DIR]
```

Example:
```shell
python demo.py --data_dir=data --visualize=False --output_dir=output
```

While visualizing, press `space` to start/pause fusion, `N` to step and `Q` to quit.

**Note**: The dataset used in this repository is a modified version of 'RedKitchen' from the original 7-Scenes dataset. The original dataset can be found [here](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

## Acknowledgements
- [TSDF Fusion in Python](https://github.com/andyzeng/tsdf-fusion-python) 
- [Open3D](https://www.open3d.org/)
- [PyTorch](https://pytorch.org/)
