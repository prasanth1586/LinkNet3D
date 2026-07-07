
# 🔷 LinkNet3D: A 3D-OD Model

This repository provides the **LinkNet3D** model implementation.

This repo is built from the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 3D object detection framework(Apache 2.0 License). We thank the team of OpenPCDet for their contribution.

LinkNet3D is designed to be integrated **without changing any core file names** — it can easily be inserted directly inside the BEV backbone module.

---
## 📊 KITTI Benchmark Results

| **Benchmark**              | **Easy** | **Moderate** | **Hard** |
|-----------------------------|----------|--------------|----------|
| Car (3D Detection)          | 87.22 %  | 78.54 %      | 74.36 %  |
| Car (Bird's Eye View)       | 91.71 %  | 88.16 %      | 84.85 %  |
| Cyclist (3D Detection)      | 76.11 %  | 60.47 %      | 53.77 %  |
| Cyclist (Bird's Eye View)   | 79.47 %  | 64.64 %      | 57.99 %  |

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `2DCRM.py` | The LinkNet3D backbone class |
| `LinkNet3D.yaml` | Configuration for training on the KITTI dataset |
| `README.md` | Setup and usage instructions |

---

## ⚙️ Setup Instructions

## Usage

1. Install OpenPCDet.
2. Replace the appropriate BEV backbone with `2DCRM.py`.
3. Register the backbone in the corresponding `__init__.py`.
4. Copy `LinkNet3D.yaml` into the appropriate `cfgs` directory.
5. Train or evaluate using the provided configuration.

This implementation assumes familiarity with the OpenPCDet framework.



## 🧠 Acknowledgements

- 🚗 Framework: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- 📄 Paper Inspiration: [SECOND: Sparsely Embedded Convolutional Detection](https://arxiv.org/abs/1811.10092)

---

## 📄 License

This project inherits the license of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache 2.0).

---
