<p align="center">
  <h1 align="center">Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration</h1>
  <p align="center">
    <a href="#"><strong>Anonymous Author 1</strong></a>
    ·
    <a href="#"><strong>Anonymous Author 2</strong></a>
    ·
    <a href="#"><strong>Anonymous Author 3</strong></a>
    ·
    <a href="#"><strong>Anonymous Author 4</strong></a>
    <br>
    <a href="#"><strong>Anonymous Author 5</strong></a>
    ·
    <a href="#"><strong>Anonymous Author 6</strong></a>
    ·
    <a href="#"><strong>Anonymous Author 7</strong></a>
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
- **GPU**: NVIDIA RTX 3090 / A100
- **Compiler**: GCC 6.3.0
- **Framework**: PyTorch 1.10+

Follow the [COSeg installation guide](https://github.com/ZhaochongAn/COSeg?tab=readme-ov-file#environment) for detailed setup.

## 📦 Dataset Preparation

### Pretraining Stage Data
We follow the MM-FSS protocol. [cite_start]You can directly download the ScanNet 3D dataset and 2D features for pretraining[cite: 328, 329]:

```bash
# Download ScanNet 3D dataset
wget [https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip)
unzip scannet_3d.zip

# Download 2D features
wget [https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip](https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip)
unzip scannet_multiview_lseg.zip
```

Link the data folder:
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
[cite_start]Our DA-FSS efficiently reuses the pre-trained alignment from the baseline (MM-FSS) to avoid extra overhead[cite: 149].

**Option A**: Download our pretrained weights [here](https://drive.google.com/drive/u/1/folders/1JoeAXJh1AZM3bM0KGBJQsFTad6uqpzUJ).

**Option B**: Train from scratch:
```bash
cd pretraining
bash run/distill_strat.sh PATH_to_SAVE_BACKBONE config/scannet/ours_lseg_strat.yaml
```

### 2. Meta-learning Stage (DA-FSS)
Train the Decoupled-experts Arbitration model. [cite_start]We freeze the backbone to maintain stability[cite: 330].

**Key Settings**:
- [cite_start]`SAM Layers`: N=1 (S3DIS), N=2 (ScanNet) [cite: 520]
- [cite_start]`DAM Weights`: $\lambda_{PLR}=0.001$, $\lambda_{DCR}=0.5$ [cite: 521]

```bash
# For 1-way tasks (Example: S3DIS)
python3 main_fs.py --config config/s3dis_DAFSS.yaml \
    save_path logs/s3dis/1w1s \
    pretrain_backbone checkpoints/backbone_weights.pth \
    cvfold 0 \
    n_way 1 \
    k_shot 1 \
    num_episode_per_comb 1000

# For 2-way tasks (Example: ScanNet)
python3 main_fs.py --config config/scannet_DAFSS.yaml \
    save_path logs/scannet/2w5s \
    pretrain_backbone checkpoints/backbone_weights.pth \
    cvfold 0 \
    n_way 2 \
    k_shot 5 \
    num_episode_per_comb 100
```

## 📊 Evaluation

[cite_start]We strictly follow the N-way K-shot episodic evaluation protocol[cite: 506].

```bash
python3 main_fs.py --config config/scannet_DAFSS.yaml \
    test True \
    eval_split test \
    weight logs/scannet/1w1s/best_model.pth \
    n_way 1 \
    k_shot 1
```



## 🎯 Model Zoo

| Model | Dataset | CVFOLD | N-way K-shot | Weights |
|:-------:|:---------:|:--------:|:------------:|:----------:|
| dafss_s30_1w1s | S3DIS | 0 | 1-way 1-shot | [Download](#) |
| dafss_s30_1w5s | S3DIS | 0 | 1-way 5-shot | [Download](#) |
| dafss_sc0_1w1s | ScanNet | 0 | 1-way 1-shot | [Download](#) |
| dafss_sc0_2w5s | ScanNet | 0 | 2-way 5-shot | [Download](#) |

## 📧 Contact

For any questions, feel free to reach out:
- **Email**: anzhaochong@outlook.com
