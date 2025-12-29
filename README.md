<p align="center">
  <h1 align="center">Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration</h1>
  <p align="center">
    <a href="#"><strong>Anonymous Author 1</strong></a>
    ·
    <a href="#"><strong>Anonymous Author 2</strong></a>
    ·
  </p>
  <h2 align="center">CVPR/ICLR 2025 Submission (<a href="https://anonymous.4open.science/r/DA-FSS-765F/">Paper</a>)</h2>
</p>

<p align="center">
  <img src="assets/figure2_architecture.png" alt="Overview of DA-FSS" width="90%">
</p>

## 🌟 Highlights

We identify a critical **"Plasticity-Stability Dilemma"** in existing "Fuse-then-Refine" multimodal paradigms, where semantic gradients dominate geometric adaptation. To address this, we present **DA-FSS**:

- [cite_start]**Decoupled-experts Arbitration**: We propose a novel architecture that physically separates geometric adaptation (Plasticity) and semantic preservation (Stability) into **Parallel Experts**[cite: 37, 42].
- [cite_start]**Gradient Harmony**: We introduce the **Decoupled Alignment Module (DAM)**, utilizing Prototype Loss Regularization (PLR) and Decoupled Consistency Regularization (DCR) to align experts without propagating confusion noise[cite: 43, 372, 386].
- [cite_start]**Smart Arbitration**: A **Stacked Arbitration Module (SAM)** is designed to effectively fuse the decoupled pathways using boundary-injected guidance, preventing the suppression of structural details[cite: 41, 403].
- [cite_start]**SOTA Performance**: DA-FSS outperforms the strong baseline (MM-FSS) on **S3DIS** and **ScanNet** datasets while using fewer parameters (-0.27M) and FLOPs (-0.30G)[cite: 531, 567].

## 📝 Citation

If you find our code or paper useful, please cite:

```bibtex
@inproceedings{dafss2025rethinking,
  title={Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration},
  author={Anonymous Authors},
  booktitle={Under Review},
  year={2025}
}
```

---

## 🛠️ Environment Setup

Our environment has been tested on:
- **GPU**: NVIDIA RTX 3090s
- **Compiler**: GCC 6.3.0
- **Framework**: PyTorch 1.10+

Follow the [COSeg installation guide](https://github.com/ZhaochongAn/COSeg?tab=readme-ov-file#environment) for detailed setup.

## 📦 Dataset Preparation

### Pretraining Stage Data
Follow [OpenScene](https://github.com/pengsongyou/openscene?tab=readme-ov-file#data-preparation) instructions, you can 
directly download the following ScanNet 3D dataset and 2D features for pretraining:
```bash
# Download ScanNet 3D dataset
wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip
unzip scannet_3d.zip

# Download 2D features
wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip
unzip scannet_multiview_lseg.zip
```

You should put the unpacked data into the folder ./pretraining/data/ or link to the corresponding data folder with the symbolic link:
```bash
ln -s /PATH/TO/DOWNLOADED/FOLDER ./pretraining/data
```

### Few-shot Stage Data
[cite_start]We use the standard S3DIS and ScanNet few-shot splits[cite: 480].

| Dataset | Download Link |
|:-------:|:-------------:|
| S3DIS | [Download Processed Data](https://drive.google.com/file/d/1frJ8nf9XLK_fUBG4nrn8Hbslzn7914Ru/view?usp=drive_link) |
| ScanNet | [Download Processed Data](https://drive.google.com/file/d/19yESBZumU-VAIPrBr8aYPaw7UqPia4qH/view?usp=drive_link) |

Ensure your `.yaml` config file points to `[PATH_to_DATASET]/blocks_bs1_s1/data`.


## 🔄 Training Pipeline

### 1. Backbone and IF Head Pretraining

**Option A**: Download our pretrained weights from [Google Drive](https://drive.google.com/drive/u/1/folders/1JoeAXJh1AZM3bM0KGBJQsFTad6uqpzUJ)

**Option B**: Train from scratch:
```bash
cd pretraining
bash run/distill_strat.sh PATH_to_SAVE_BACKBONE config/scannet/ours_lseg_strat.yaml


### 2. Meta-learning Stage
Set config `config/[CONFIG_FILE]` to be `s3dis_COSeg_fs.yaml` or `scannetv2_COSeg_fs.yaml` for training on S3DIS or ScanNet respectively.
Adjust `cvfold`, `n_way`, and `k_shot` according to your few-shot task:

```bash
# For 1-way tasks
python3 main_fs.py --config config/[CONFIG_FILE] \
    save_path [PATH_to_SAVE_MODEL] \
    pretrain_backbone [PATH_to_SAVED_BACKBONE] \
    cvfold [CVFOLD] \
    n_way 1 \
    k_shot [K_SHOT] \
    num_episode_per_comb 1000

# For 2-way tasks
python3 main_fs.py --config config/[CONFIG_FILE] \
    save_path [PATH_to_SAVE_MODEL] \
    pretrain_backbone [PATH_to_SAVED_BACKBONE] \
    cvfold [CVFOLD] \
    n_way 2 \
    k_shot [K_SHOT] \
    num_episode_per_comb 100
```

> **Note**: Following [COSeg](https://github.com/ZhaochongAn/COSeg?tab=readme-ov-file#training-pipeline), `num_episode_per_comb` defaults to 1000 for 1-way and 100 for 2-way tasks to maintain consistency in test set size.

### Model Evaluation
Modify `cvfold`, `n_way`, `k_shot` and `num_episode_per_comb` accordingly and run:
```bash
python3 main_fs.py --config config/[CONFIG_FILE] \
    test True \
    eval_split test \
    weight [PATH_to_SAVED_MODEL] \
    [vis 1]  # Optional: Enable W&B visualization
```

> **Note**: Performance may vary by 1.0% due to potential randomness in the training process. ScanNetv2 typically shows less variance than S3DIS.

### Visualization
Follow [COSeg visualization guide](https://github.com/ZhaochongAn/COSeg?tab=readme-ov-file#visualization) for high-quality visualization results.


## 🎯 Model Zoo

| Model | Dataset | CVFOLD | N-way K-shot | Weights |
|:-------:|:---------:|:--------:|:------------:|:----------:|
| s30_1w1s | S3DIS | 0 | 1-way 1-shot | [Download]() |
| s30_1w5s | S3DIS | 0 | 1-way 5-shot | [Download]() |
| s30_2w1s | S3DIS | 0 | 2-way 1-shot | [Download]() |
| s30_2w5s | S3DIS | 0 | 2-way 5-shot | [Download]() |
| s31_1w1s | S3DIS | 1 | 1-way 1-shot | [Download]() |
| s31_1w5s | S3DIS | 1 | 1-way 5-shot | [Download]() |
| s31_2w1s | S3DIS | 1 | 2-way 1-shot | [Download]() |
| s31_2w5s | S3DIS | 1 | 2-way 5-shot | [Download]() |
| sc0_1w1s | ScanNet | 0 | 1-way 1-shot | [Download]() |
| sc0_1w5s | ScanNet | 0 | 1-way 5-shot | [Download]() |
| sc0_2w1s | ScanNet | 0 | 2-way 1-shot | [Download]() |
| sc0_2w5s | ScanNet | 0 | 2-way 5-shot | [Download]() |
| sc1_1w1s | ScanNet | 1 | 1-way 1-shot | [Download]() |
| sc1_1w5s | ScanNet | 1 | 1-way 5-shot | [Download]() |
| sc1_2w1s | ScanNet | 1 | 2-way 1-shot | [Download]() |
| sc1_2w5s | ScanNet | 1 | 2-way 5-shot | [Download]() |

## 📧 Contact

For any questions, feel free to reach out:
