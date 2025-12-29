<p align="center">
  <h1 align="center">Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration</h1>
  <p align="center">
    <a href="#"><strong>Anonymous Author 1</strong></a>
    ·
    <a href="#"><strong>Anonymous Author 2</strong></a>
    ·
    <a href="#"><strong>Anonymous Author 3</strong></a>
    <br>
    <strong>CVPR/ICLR Submission 202X</strong>
  </p>
  
  <h2 align="center">Paper (<a href="https://anonymous.4open.science/r/DA-FSS-765F/">Code Available</a>)</h2>
  
  <div align="center">
    <img src="https://img.shields.io/badge/Task-Few--Shot_Segmentation-red.svg" alt="Task">
    <img src="https://img.shields.io/badge/Modality-3D_Point_Cloud_+_Language-blue.svg" alt="Modality">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" alt="Framework">
  </div>
</p>

<p align="center">
  <img src="assets/figure2_architecture.png" alt="Overview of DA-FSS" width="95%">
</p>

## 🌟 Highlights

[cite_start]We identify a critical **"Plasticity-Stability Dilemma"** in existing "Fuse-then-Refine" multimodal paradigms, where semantic gradients dominate geometric adaptation[cite: 35, 62]. To address this, we present **DA-FSS**:

- [cite_start]**Decoupled-experts Arbitration**: We propose a novel architecture that physically separates geometric adaptation (Plasticity) and semantic preservation (Stability) into **Parallel Experts**[cite: 37, 42].
- [cite_start]**Gradient Harmony**: We introduce the **Decoupled Alignment Module (DAM)**, utilizing Prototype Loss Regularization (PLR) and Decoupled Consistency Regularization (DCR) to align experts without propagating confusion noise[cite: 43, 372, 381].
- [cite_start]**Smart Arbitration**: A **Stacked Arbitration Module (SAM)** is designed to effectively fuse the decoupled pathways using boundary-injected guidance, preventing the suppression of structural details[cite: 41, 402].
- [cite_start]**SOTA Performance**: DA-FSS outperforms the strong baseline (MM-FSS) on **S3DIS** and **ScanNet** datasets while using fewer parameters and FLOPs[cite: 44, 567].

## 📝 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{da_fss_2025,
  title={Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration},
  author={Anonymous Authors},
  booktitle={Under Review},
  year={2025}
}

🛠️ Environment SetupOur environment has been tested on:GPU: NVIDIA RTX 3090Compiler: GCC 6.3.0Framework: PyTorch 1.10+Follow the COSeg installation guide for detailed setup.📦 Dataset PreparationPretraining Stage DataWe follow the MM-FSS protocol. You can directly download the ScanNet 3D dataset and 2D features for pretraining:Bash# Download ScanNet 3D dataset
wget [https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip)
unzip scannet_3d.zip

# Download 2D features
wget [https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip](https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip)
unzip scannet_multiview_lseg.zip
Link the data folder:Bashln -s /PATH/TO/DOWNLOADED/FOLDER ./pretraining/data
Few-shot Stage DataWe use the standard S3DIS and ScanNet few-shot splits.DatasetDownload LinkS3DISDownload Processed DataScanNetDownload Processed DataEnsure your .yaml config file points to [PATH_to_DATASET]/blocks_bs1_s1/data.🔄 Training Pipeline1. Backbone and IF Head PretrainingOur DA-FSS efficiently reuses the pre-trained alignment from the baseline (MM-FSS)1.Option A: Download our pretrained weights here.Option B: Train from scratch:Bashcd pretraining
bash run/distill_strat.sh PATH_to_SAVE_BACKBONE config/scannet/ours_lseg_strat.yaml
2. Meta-learning Stage (DA-FSS)Train the Decoupled-experts Arbitration model. We freeze the backbone to maintain stability.Key Settings:SAM Layers: N=1 (S3DIS), N=2 (ScanNet) 2DAM Weights: $\lambda_{PLR}=0.001$, $\lambda_{DCR}=0.5$ 3Bash# For 1-way tasks (Example: S3DIS)
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
📊 EvaluationWe strictly follow the N-way K-shot episodic evaluation protocol4.Bashpython3 main_fs.py --config config/scannet_DAFSS.yaml \
    test True \
    eval_split test \
    weight logs/scannet/1w1s/best_model.pth \
    n_way 1 \
    k_shot 1
