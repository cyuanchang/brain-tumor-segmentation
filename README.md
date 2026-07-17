# Brain Tumor Segmentation

Documentary repository for a Veritas AI Capstone project: **glioma tumor segmentation from MRI** with a baseline U-Net, plus patient-level tumor volume estimation.

**Team:** Esha Verma, Deeksha, Wellon Chen, Victor Wong — 8/13/2023

---

## Motivation

Brain tumors are a deadly disease. Accurately detecting tumors matters for treatment and prognosis. This project segments **glioma** tumors from MRI using deep learning.

---

## Data

Derived from **BraTS 2018**-style slices prepared for the Capstone:

| | |
|---|---|
| MRI slices | **4,715** |
| Patients with glioma | **152** |

Each sample is an MRI slice (`.jpg`) paired with a segmentation mask (`.png`).

Expected layout for local training:

```text
data/
  train/images/*.jpg
  train/masks/*.png
  val/images/*.jpg
  val/masks/*.png
```

---

## Exploratory Data Analysis

Example slice statistics from the Capstone presentation:

**MRI**

- Min = 0, Max = 249, Mean = 24, Std = 45

**Mask**

- Min = 0, Max = 150, Mean = 4, Std = 21
- Unique pixel values: `0, 50, 100, 150` (background + multi-region tumor labels)

The baseline model **binarizes** masks: any non-zero label → tumor (class 1).

---

## Preprocessing

Pipeline steps (from the project presentation / Colab):

1. Open  
2. Decode  
3. Binarize classes  
4. Resize → `(120, 120)`  
5. One-hot encode  

Implemented in `src/brain_tumor_segmentation/data.py` as `preprocess_data` and `create_data_pipeline` (`tf.data`, shuffle / batch / prefetch). By default the Colab used a **500-slice** subset for faster iteration; use `--full-dataset` to disable that cap.

---

## Baseline Model

A **2-level U-Net** (encoder → bottleneck → decoder with skip connections), trained with categorical cross-entropy and a tumor Dice metric.

Default settings (from Colab):

| Setting | Value |
|---|---|
| Input shape | `(120, 120, 1)` |
| Classes | 2 (background / tumor) |
| Batch size | 16 |
| Epochs | 20 |
| Optimizer | Adam |

Model code: `src/brain_tumor_segmentation/models/unet.py`

---

## Results

Patient-level **predicted vs ground-truth tumor volumes** (slice area × 3.5 mm thickness → mL):

- Linear fit: **y = 1.09x − 1.41**, **r = 0.76**
- Bland–Altman mean difference ≈ **−0.21**

Interpretation from the presentation: performance is reasonable, but **larger tumors tend to be predicted larger than they really are**.

---

## Summary & Future Work

**Done**

- Initial U-Net for MRI glioma segmentation and volume estimation  
- Volume correlation **r ≈ 0.76**

**Next**

- Train longer (more epochs)  
- Try additional architectures for higher accuracy  

---

## Repository layout

```text
brain_tumor_segmentation/
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   └── original_colab.ipynb
├── scripts/
│   ├── train.py
│   └── evaluate.py
└── src/brain_tumor_segmentation/
    ├── config.py
    ├── data.py
    ├── metrics.py
    ├── evaluate.py
    ├── volume.py
    ├── viz.py
    └── models/unet.py
```

Code is cleaned from the Capstone [Colab notebook](https://colab.research.google.com/drive/1ZYrTpLoDivjDKqAazFdhRYUrNaoHQ-0u). Project narrative is summarized in this README.

---

## Setup

```bash
cd E:\brain_tumor_segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Or: `pip install -r requirements.txt` and set `PYTHONPATH=src`.

GPU (optional) follows your TensorFlow / CUDA install.

---

## Train

```bash
python scripts/train.py --data-root path\to\BraTS_split --epochs 20
```

Useful flags:

- `--full-dataset` — use all slices (no 500-sample cap)  
- `--output-dir artifacts` — where `baseline_unet.keras` is saved  

---

## Evaluate

```bash
python scripts/evaluate.py --data-root path\to\BraTS_split --model-path artifacts\baseline_unet.keras
```

Prints mean tumor Dice on the validation subset and Pearson **r** for patient volumes (same analysis as the Colab).

---

## Original sources

- Colab: https://colab.research.google.com/drive/1ZYrTpLoDivjDKqAazFdhRYUrNaoHQ-0u  
- Notebook snapshot: `notebooks/original_colab.ipynb`  
