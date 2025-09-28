# Model Directory

This folder contains the network architectures used in the paper **E2SRLF: End-to-End Light Field Image Super-Resolution Depth Estimation**.
It includes the main model and several ablation variants for comparison and analysis.

## Files

* **`model.py`**
  The proposed **E2SRLF** model.

  * End-to-end network that directly generates **high-resolution disparity maps** from low-resolution light field images.
  * Includes all proposed modules: **MDCAT (SACAT + DCAT)**, **SSFU**, and **HLLoss**.

* **`model_x1.py`**
  The **non-super-resolution version** of E2SRLF.

  * Used to verify that the proposed mechanisms (MDCAT, HLLoss) remain effective even without SR.
  * Serves as the **baseline comparison** for LR-only disparity estimation.

* **`model_NAT.py`**
  Ablation model **without any attention mechanisms**.

  * Used to evaluate the impact of MDCAT (attention modules) on feature extraction.
  * Compared against E2SRLF and SACAT-only variants in the ablation study.

* **`model_SACAT.py`**
  Ablation model with **SACAT (Spatial-Angular Channel Attention)** only.

  * Helps analyze the contribution of SACAT compared to DCAT and the full MDCAT.
  * Serves as an intermediate configuration between `model_NAT.py` and `model.py`.

---

## Usage

You can specify which model to use by passing the corresponding network file to the training script, e.g.:

```bash
# Train the main E2SRLF model
python implement/implement.py --net model --mode train --device cuda:0

# Train the non-SR variant (x1)
python implement/implement.py --net model_x1 --mode train --device cuda:0

# Train the ablation model without attention
python implement/implement.py --net model_NAT --mode train --device cuda:0

# Train the ablation model with SACAT only
python implement/implement.py --net model_SACAT --mode train --device cuda:0
```

---

## Notes

* **`model.py`** = Full E2SRLF (SR + MDCAT + SSFU + HLLoss)
* **`model_x1.py`** = E2SRLF without SR (scale factor = 1)
* **`model_NAT.py`** = E2SRLF without attention modules
* **`model_SACAT.py`** = E2SRLF with only SACAT enabled

These models are used in the **comparison (Table I, II)** and **ablation studies (Table III, IV)** of the paper.