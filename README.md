# GraspNet-3D — Screw Segmentation for Robotic Grasping

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Open3D](https://img.shields.io/badge/Open3D-3D-00A6D6?style=for-the-badge&logoColor=white)](www.open3d.org)
[![PointNet++](https://img.shields.io/badge/PointNet%2B%2B-Segmentation-8A2BE2?style=for-the-badge&logoColor=white)](https://arxiv.org/abs/1706.02413)

A 3D deep-learning pipeline that segments a single screw from noisy point clouds, built to support precise robotic grasping in cluttered scenes. Covers the full workflow: raw point-cloud preprocessing, PointNet++ training, and Open3D visualisation of segmentation results.

---

## 1. Overview

Robotic grasping needs to isolate a target object from surrounding clutter in 3D space. Raw scans arrive as variable-length, noisy point clouds with no fixed shape, which makes them awkward to feed straight into a neural network.

This project turns those raw scans into a trained segmentation model that labels every point as object or background, then visualises the result in 3D so the output can be inspected rather than just trusted. It is built on the **PointNet++ semantic-segmentation** architecture and trained in **PyTorch**.

> **Reported accuracy.** The project documentation reports **99.01%** accuracy for the trained model. This figure is as reported in the original documentation and has not been independently verified.

## 2. Problem

Three things make 3D screw segmentation hard for a grasping pipeline:

- **Variable-length input.** A single scan can hold over 200,000 points, and the count changes from scan to scan.
- **Noise and clutter.** The target screw sits among background points that must be rejected, not averaged in.
- **Fixed-shape networks.** Neural networks expect a consistent input tensor, so raw clouds must be reshaped without discarding geometry.

## 3. Pipeline

```
Raw point clouds (.txt)
        │  variable length, x y z label nx ny nz
        ▼
[1. Preprocessing]   downsample to 8192 points, save .npz
        │
        ▼
[2. Training]        PointNet++ semantic segmentation (PyTorch)
        │
        ▼
Trained model (.pth)
        │
        ▼
[3. Inference]       per-point labels → Open3D 3D visualisation
```

### 3.1 Dataset

**Raw format (`.txt`)** — each file is a variable-length point cloud (for example ~221,928 points). One point per line:

```
x, y, z, label, nx, ny, nz
```

| Field | Meaning |
|-------|---------|
| `x, y, z` | Point coordinates |
| `label` | `0` = background, `1` = screw |
| `nx, ny, nz` | Surface normals |

### 3.2 Preprocessing (`1-preprocessing.ipynb`)

Raw clouds are converted to a fixed size so they can be batched:

- **Downsampled to 8192 points** per sample.
- Saved as `.npz` files (`sample_000.npz` … `sample_297.npz`), each containing:
  - `points` — shape `(8192, 6)`, the `[x, y, z, nx, ny, nz]` features
  - `labels` — shape `(8192,)`, the per-point class labels
- A `ScrewSegTxtDataset` (PyTorch `Dataset`) loads the raw `.txt` files, samples a fixed number of points, and normalises coordinates.
- A visualisation helper (`visualize_sample`) renders a sample point cloud coloured by label so preprocessing can be checked by eye.

### 3.3 Model

- **Architecture:** PointNet++ semantic segmentation (`models/pointnet2_sem_seg.py`, loaded via `get_model`)
- **Input:** `[B, 6, N]` — batch of `N` points with 6 features (xyz + normals)
- **Output:** `[B, N, 2]` — per-point logits over 2 classes
- **Classes:** background (`0`), screw (`1`)

### 3.4 Training (`2-training.ipynb`)

| Setting | Value |
|---------|-------|
| Framework | PyTorch |
| Dataset loader | `PreprocessedScrewDataset` (reads the `.npz` files, optional augmentation) |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 16 |
| Epochs | 20 |

The trained checkpoint is saved as `pointnet2_screw_segmentation_final.pth`.

### 3.5 Inference & Visualisation (`3-inference-open3d.ipynb`)

- Loads a preprocessed `.npz` sample and formats it as `[1, 6, 8192]`.
- Runs the trained model and takes the predicted label per point.
- Visualises the result in **Open3D**, colouring:
  - **background** grey `[0.6, 0.6, 0.6]`
  - **screw** red `[1.0, 0.0, 0.0]`
- Example output: **969 screw points detected** in one sample.

![Screw Segmentation Output](Open3D%20videos/Image.png)

Demo visualisations are included under [`Open3D%20videos/`](Open3D%20videos/).

## 4. Results

- **End-to-end pipeline** from raw point cloud → trained model → 3D segmentation visualisation.
- **Reported accuracy: 99.01%** (as stated in the project documentation; not independently verified).
- Example inference detected **969 screw points** in a single sample.

## 5. Repository Structure

```
graspnet-3d/
├── 1-preprocessing.ipynb                    # Raw .txt → fixed-size .npz
├── 2-training.ipynb                         # PointNet++ training loop
├── 3-inference-open3d.ipynb                 # Inference + Open3D visualisation
├── models/                                  # PointNet++ semantic-segmentation model
├── Screw_Segmentation_Documentation.pdf     # Full project documentation
├── pointnet2_screw_segmentation_final.pth   # Trained model weights
└── Open3D videos/                           # Demo visualisations (videos + screenshot)
```

## 6. Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- Open3D, NumPy, Matplotlib, tqdm

```
pip install torch numpy matplotlib tqdm open3d
```

### Run the pipeline

Open the notebooks in order — each reads the saved output of the previous one, so nothing is recomputed silently:

1. **`1-preprocessing.ipynb`** — convert raw `.txt` point clouds to fixed-size `.npz` samples.
2. **`2-training.ipynb`** — train PointNet++ on the preprocessed data and save the checkpoint.
3. **`3-inference-open3d.ipynb`** — run inference on a sample and view the coloured segmentation in Open3D.

## 7. Limitations

Trained and evaluated on a **single-object screw dataset**, so results do not generalise to other objects or multi-object scenes. The 99.01% figure reflects that specific dataset and setup.

## 8. Author

- **GitHub:** [@tzmughal](https://github.com/tzmughal)

## License

Add a license before extending or redistributing this repository.
