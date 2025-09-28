# Implement Directory

This folder contains the **training, validation, testing scripts, and utility functions** for the paper:
**E2SRLF: End-to-End Light Field Image Super-Resolution Depth Estimation**.

---

## Files

* **`implement.py`**
  The **main entry point** for training, validation, and testing.

  * Supports different modes: `train`, `valid`, `test`.
  * Loads the corresponding model (`model.py`, `model_x1.py`, `model_NAT.py`, `model_SACAT.py`).
  * Saves results to the `Results/` folder and checkpoints to `param/`.

* **`train.py`**
  Standard training script for **E2SRLF**.

  * Trains the main **super-resolution model (model.py)**.
  * Uses all proposed modules: MDCAT, SSFU, HLLoss.

* **`train_x1.py`**
  Training script for the **non-SR variant (model_x1.py)**.

  * Disparity estimation only at input resolution (no SR).
  * Used for baseline comparison.

* **`train_SRL1.py`**
  Training script for **E2SRLF with SRL1 loss** variant.

  * Alternative training configuration for ablation studies.
  * Helps analyze the impact of HLLoss vs. SRL1.

* **`valid.py`**
  Validation script.

  * Loads pretrained checkpoints.
  * Evaluates the model on the validation set and reports metrics (MSE, PSNR, SSIM).

* **`test.py`**
  Testing script.

  * Runs inference on test datasets.
  * Saves disparity results in `.pfm` format for quantitative evaluation.

* **`utils.py`**
  Utility functions for:

  * **Data loading & preprocessing**
  * **Data augmentation** (flip, rotate, brightness, contrast)
  * Helper functions used by training/validation loops

* **`func_pfm.py`**
  Helper for handling `.pfm` files (Portable Float Map).

  * Reading and writing disparity maps in **.pfm** format.
  * Ensures compatibility with the **HCI 4D Light Field Benchmark**.

---

## Usage

### Train

```bash
# Train the main E2SRLF model
python implement/implement.py --net E2SRLF --mode train --device cuda:0

# Train the non-SR variant
python implement/implement.py --net E2SRLF_x1 --mode train --device cuda:0

# Train NAT ablation model (no attention)
python implement/implement.py --net E2SRLF_NAT --mode train --device cuda:0

# Train SACAT ablation model
python implement/implement.py --net E2SRLF_SACAT --mode train --device cuda:0

# Train SRL1 ablation model
python implement/implement.py --net E2SRLF_SRL1 --mode train --device cuda:0
```

### Validation

```bash
python implement/implement.py --net E2SRLF --mode valid --device cuda:0
```

### Test

```bash
python implement/implement.py --net E2SRLF --mode test --device cuda:0
```

The output disparity maps will be saved as `.pfm` files in `./Results/'NetName'`.

---

## Notes

* Use **`utils.py`** for dataset preprocessing and augmentation before training.
* **`func_pfm.py`** ensures correct handling of `.pfm` format results for benchmark evaluation.
* Different training scripts (`train.py`, `train_x1.py`, `train_SRL1.py`) correspond to different experimental configurations from the paper (main model vs. ablation studies).
