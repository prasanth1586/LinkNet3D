
# 🔷 LinkNet3D: A 3D-OD Model

This repository provides the **LinkNet3D** backbone implementation for use within the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 3D object detection framework(Apache 2.0 License).

LinkNet3D is designed to be integrated **without changing any core file names** — it is inserted directly inside the BEV backbone module.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `linknet3d.py` | The LinkNet3D backbone class |
| `linknet3d.yaml` | Configuration for training on the KITTI dataset |
| `README.md` | Setup and usage instructions |

---

## ⚙️ Setup Instructions

### ✅ Step 1: Clone OpenPCDet

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
```
Install dependencies and compile CUDA ops:
---

## 📂 Dataset Preparation

```bash
Download the official **KITTI 3D Object Detection** dataset from https://www.cvlibs.net/datasets/kitti/index.php and organize it as follows
LinkNet3D
├── OpenPCDet
|    ├── data
|    │   ├── kitti
|    │   │   │── ImageSets
|    │   │   │── training
|    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
|    │   │   │── testing
|    │   │   │   ├──calib & velodyne & image_2
|    ├── pcdet
|    ├── tools
```

### 🔧 Step 2: Add LinkNet3D Backbone
1. Copy `linknet3d.py` to:

```
OpenPCDet/pcdet/models/backbones_2d/bev_backbone/
```

2. Open this file for editing:

```
OpenPCDet/pcdet/models/backbones_2d/bev_backbone/__init__.py
```

3. Add this line at the top (with other imports):

```python
from .linknet3d import LinkNet3D
```

> ✅ This will register the class so that it can be used in your config file.

---
Note: This implementation assumes basic familiarity with OpenPCDet and 3D object detection workflows. LinkNet3D is built on OpenPCDet’s existing codebase.

### 📝 Step 3: Add the Config File

Copy `linknet3d.yaml` to:

```
OpenPCDet/tools/cfgs/kitti_models/
```

You can now train using this config.

---

## 🚀 Training

Run training using:

```bash
python train.py --cfg_file cfgs/kitti_models/linknet3d.yaml
```

You can add arguments like:

```bash
python train.py --cfg_file cfgs/kitti_models/linknet3d.yaml --epochs 80 --workers 4
```

---

## 🧪 Evaluation

After training is complete, evaluate the checkpoint:

```bash
python test.py --cfg_file cfgs/kitti_models/linknet3d.yaml --ckpt <path_to_your_checkpoint.pth>
```

---

## 📊 TensorBoard

To monitor training:

```bash
tensorboard --logdir=output
```

Logged metrics include:
- `loss`
- `learning rate`

You can optionally log `accuracy` by modifying the training loop in `train_utils/train_utils.py`.

---

## 🧠 Acknowledgements

- 🚗 Framework: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- 📄 Paper Inspiration: [SECOND: Sparsely Embedded Convolutional Detection](https://arxiv.org/abs/1811.10092)

---

## 🏁 Results (Placeholder)

KITTI benchmark results below once available._



---

## 📄 License

This project inherits the license of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache 2.0).

---

## 🙋‍♂️ Author

**Your Name**  
GitHub: [@your-username](https://github.com/your-username)  
Email: your.email@example.com
