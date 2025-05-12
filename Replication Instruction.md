
# LBUNet with AttentionSkipFusion

## 📖 Overview
This repository contains the implementation of the baseline LBUNet and a modified version with AttentionSkipFusion (ASF) for medical image segmentation.

The baseline LBUNet leverages boundary-aware supervision and multiple auxiliary outputs for high segmentation accuracy. The modified model introduces AttentionSkipFusion for learnable skip connections while simplifying the decoder and removing auxiliary branches for faster inference.

---

## 🔁 Replication Instructions

Follow the steps below to train and evaluate both the baseline and modified LBUNet models:

### 📁 Step 1: Prepare Dataset on Google Drive

Create a directory in your **Google Drive** (e.g., `ISIC_2018_Dataset/`) and place the dataset inside it with the following structure:

```
ISIC_2018_Dataset/
├── train/
│   ├── images/
│   ├── masks/
│   ├── points_boundary2/
├── val/
│   ├── images/
│   ├── masks/
├── test/
│   ├── images/
│   ├── masks/
```

---

### 🛠️ Step 2: Update `dataset.py` with Dataset Directory

The default dataset path is hardcoded as:
```python
/content/drive/MyDrive/KaggleDatasets/ISIC_Datasets/ISIC_2018_Dataset/
```
Modify this in `dataset.py` or pass a `path_Data` argument in the config if your dataset path differs.

---

### 🧠 Step 3: Modify the Model Architecture

Implement your model changes (e.g., replacing merge modules with `AttentionSkipFusion`, changing supervision structure) in `lbunet.py`. Ensure the `forward()` method and output structure reflect these changes.

---

### 🔧 Step 4: Adapt Training and Utility Scripts

In `utils.py`:
- Update the loss function if the model output structure is changed.
- Use `GT_BceDiceLoss` or `BceDiceLoss` as needed.

In `engine.py`:
- Ensure `train_one_epoch()` and `val_one_epoch()` match your model’s output format.

In `train.py`:
- Ensure the correct network is selected and model config is defined in `setting_config()`.

---

### 🔗 Step 5: Mount Google Drive in Google Colab

In your Colab notebook:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 📂 Step 6: Set Working Directory and Adjust Paths

In `train.py`, the root directory is defined as:
```python
os.chdir('/content/drive/MyDrive/LBUNet Cancer')
```
Change this to your actual path if needed.

---

### 📦 Step 7: Install Required Dependencies

Install dependencies in Colab:
```bash
!pip install tensorboardX timm scikit-learn pillow
```

---

### 🚀 Step 8: Run the Training Script

Start training:
```bash
!python train.py
```
The script handles training, validation, saving the best model, and final testing.

---

## 📜 License

This project is for research and academic use only.

## ✏️ Acknowledgment

This work uses OpenAI’s ChatGPT to assist in formalizing, summarizing, and organizing parts of the code and documentation.

