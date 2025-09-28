# E2SRLF: End-to-End Light Field Image Super-Resolution Depth Estimation

## Overview

E2SRLF (**End-to-End Super-Resolution Light Field Depth Estimation Network**) is a novel end-to-end framework that can **directly generate high-resolution disparity maps (HR disparity maps) from low-resolution light field images (LR LF)**.

Unlike traditional methods that only estimate disparity at the same resolution as the input, E2SRLF addresses this challenge with the following innovations:

* **Multi-Dimensional Channel Attention Mechanism (MDCAT)**
  Combining **SACAT** (Spatial-Angular Channel Attention) and **DCAT** (Disparity Channel Attention) for accurate multi-dimensional feature weighting.

* **Spatial Super-Resolution Fusion Upsampling (SSFU)**
  Constructs a super-resolution dimension in the cost volume and fuses it with spatial information, enabling the generation of more accurate HR disparity information.

* **High-Low Resolution Collaborative Constraint Loss (HLLoss)**
  Introduces joint HR and LR constraints during training to enhance generalization and robustness.

The **network architecture** is illustrated in **Fig.2**.

---

## Highlights

* Directly generates **HR disparity maps from LR light field images**.
* Incorporates **MDCAT attention** for stronger global and local feature representation.
* **SSFU** enables interaction between spatial and disparity dimensions, improving SR accuracy.
* **HLLoss** ensures learning stability by enforcing constraints at both HR and LR levels.

**Comparison results** (Fig.1, Fig.5, Fig.6, Fig.7, Fig.8) show that E2SRLF outperforms existing methods on both synthetic and real-world datasets, particularly in fine detail preservation and occlusion handling.

---

## Requirement

* PyTorch >= 1.13.0, torchvision >= 0.15.0
* Python = 3.8, CUDA = 11.0
* A GPU with sufficient memory
* Disparity range is [-4, 4], corresponding to dilation rate = 9

---

## Datasets

* **Training & Validation**: [HCI 4D Light Field Benchmark](https://lightfield-analysis.uni-konstanz.de/)
* **Preprocessing**: Images of size `512×512` are downsampled to `256×256` as input, while disparity labels are proportionally scaled (Eq.5 in the paper).
* Data augmentation includes random flips, rotations, brightness, and contrast adjustments.

---

### Path structure
- If not specified, it is a file name; if specified, it is a virtual category.
```
.
├── dataset
│   ├── training
│   └── validation
├── Figure
│   └── paper_picture
├── implement 
│   ├── implementation     # (virtual category, not an actual folder)
│   └── data_preprocessing # (virtual category, not an actual folder)
├── model
│   └── network_functions  # (virtual category, not an actual folder)
├── param
│   └── checkpoints        # (virtual category, not an actual folder)
└── Results
    ├── our_network        # (virtual category, not an actual folder)
    │   ├── E2SRLF
    │   └── E2SRLF_x1
    └── Analysis           # (virtual category, not an actual folder)
        ├── E2SRLF_NAT 
        ├── E2SRLF_SACAT
        └── E2SRLF_SRL1

```

---

## Train

* Modify hyper-parameters in `parse_args()` if needed. Default settings follow the paper.
* Start training:

  ```bash
  python implement/implement.py --net E2SRLF --n_epochs 10000 --mode train --device cuda:1
  ```
* Checkpoints will be saved in `./param/'NetName'`.

---

## Validation and Test

* Run with pretrained weights:

  ```bash
  python implement/implement.py --net E2SRLF --mode valid --device cuda:1
  python implement/implement.py --net E2SRLF --mode test  --device cuda:1
  ```
* Results will be saved to `./Results/'NetName'` as `scene_name.pfm`.

---

## Results

### Comparison with State-of-the-Art

* On **HCI 4D Light Field Benchmark**, E2SRLF achieves lower **MSE** and higher **PSNR/SSIM** compared to traditional depth estimation and two-stage SR methods (e.g., **SR-Distg**, **SR-MRAE**) (see Table I, II).
* **Qualitative results** (Fig.5–8) show:

  * **E2SRLF x1** achieves comparable or better performance than several existing methods under LR settings.
  * **E2SRLF** significantly outperforms two-stage methods in HR, offering sharper details and better occlusion handling.

### Ablation Studies

* **MDCAT**: Adding **SACAT** and **DCAT** sequentially leads to significant accuracy improvements (Table III, Fig.9).
* **HLLoss**: Adding LR constraints further enhances generalization and stability (Table IV, Fig.9).

---

## Citation

If you use this code or model, please cite our paper:

```
@article{E2SRLF2025,
  title={E2SRLF: End-to-End Light Field Image Super-Resolution Depth Estimation},
  author={Jie Li and Chuanlun Zhang and Xiaoyan Wang and Xinjia Li and Lin Wang and Yuxin Zeng and Yiguang Liu},
  year={2025}
}
```

