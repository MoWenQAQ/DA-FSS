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
  <img src="assets/figure2_architecture.png" alt="Overview" width="90%">
</p>

## 🌟 Highlights

We identify a critical **"Plasticity-Stability Dilemma"** in existing multimodal paradigms and propose **DA-FSS**:
- **Decoupled-experts Arbitration**: A novel architecture that physically separates geometric adaptation (Plasticity) and semantic preservation (Stability) into **Parallel Experts**
- **Gradient Harmony**: Introducing the **Decoupled Alignment Module (DAM)** with Prototype Loss Regularization (PLR) to align experts without propagating confusion noise
- **Smart Arbitration**: A **Stacked Arbitration Module (SAM)** that effectively fuses decoupled pathways using boundary-injected guidance
- **SOTA Performance**: Superior performance over MM-FSS on **S3DIS (+1.16%)** and **ScanNet (+1.00%)**, solving semantic blindness with a massive **+10.9% mAcc** gain

## 📝 Citation
If you find our code or paper useful, please cite:

```bibtex
@inproceedings{dafss2025rethinking,
  title={Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration},
  author={Anonymous Authors},
  booktitle={Under Review},
  year={2025}
}
🛠️ Environment SetupOur environment has been tested on:RTX 3090 GPUsGCC 6.3.0Follow the COSeg installation guide for detailed setup.📦 Dataset PreparationPretraining Stage DataWe strictly follow the MM-FSS protocol. You can directly download the ScanNet 3D dataset and 2D features for pretraining:Bash# Download ScanNet 3D dataset
wget [https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip)
unzip scannet_3d.zip

# Download 2D features
wget [https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip](https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip)
unzip scannet_multiview_lseg.zip
You should put the unpacked data into the folder ./pretraining/data/ or link to the corresponding data folder with the symbolic link:Bashln -s /PATH/TO/DOWNLOADED/FOLDER ./pretraining/data
Few-shot Stage DataOption 1: Direct Download (Recommended)Download our preprocessed datasets (same as Baseline):DatasetFew-shot Stage DataS3DISDownloadScanNetDownloadOption 2: Manual PreprocessingFollow COSeg preprocessing instructions.The processed data will be in [PATH_to_DATASET_processed_data]/blocks_bs1_s1/data. Make sure to update the data_root entry in the .yaml config file.🔄 Training Pipeline1. Backbone and IF Head PretrainingOption A: Download MM-FSS official weights (Recommended, as we reuse the alignment).Option B: Train from scratch:Bashcd pretraining
bash run/distill_strat.sh PATH_to_SAVE_BACKBONE config/scannet/ours_lseg_strat.yaml
2. Meta-learning Stage (DA-FSS)Set config config/[CONFIG_FILE] to be s3dis_DAFSS.yaml or scannet_DAFSS.yaml.Adjust cvfold, n_way, and k_shot according to your few-shot task:Bash# For 1-way tasks (Example: S3DIS)
python3 main_fs.py --config config/[CONFIG_FILE] \
    save_path [PATH_to_SAVE_MODEL] \
    pretrain_backbone [PATH_to_SAVED_BACKBONE] \
    cvfold [CVFOLD] \
    n_way 1 \
    k_shot [K_SHOT] \
    num_episode_per_comb 1000

# For 2-way tasks (Example: ScanNet)
python3 main_fs.py --config config/[CONFIG_FILE] \
    save_path [PATH_to_SAVE_MODEL] \
    pretrain_backbone [PATH_to_SAVED_BACKBONE] \
    cvfold [CVFOLD] \
    n_way 2 \
    k_shot [K_SHOT] \
    num_episode_per_comb 100
Note: For DA-FSS specific hyperparameters, we set SAM Layers=1 for S3DIS and SAM Layers=2 for ScanNet. PLR weight is set to 0.001.📊 Evaluation & VisualizationModel EvaluationModify cvfold, n_way, k_shot accordingly and run:Bashpython3 main_fs.py --config config/[CONFIG_FILE] \
    test True \
    eval_split test \
    weight [PATH_to_SAVED_MODEL]
Quantitative Comparison (Strict Control)DatasetSettingMM-FSS (Baseline)DA-FSS (Ours)GainS3DIS1-way 1-shot48.5049.14+0.64%S3DIS1-way 5-shot55.8456.43+0.59%ScanNet1-way 1-shot44.4645.46+1.00%ScanNet2-way 5-shot44.2045.53+1.33%VisualizationFollow COSeg visualization guide for high-quality visualization results.🎯 Model ZooModelDatasetCVFOLDN-way K-shotWeightsdafss_s30_1w1sS3DIS01-way 1-shotDownloaddafss_s30_1w5sS3DIS01-way 5-shotDownloaddafss_s30_2w1sS3DIS02-way 1-shotDownloaddafss_s30_2w5sS3DIS02-way 5-shotDownloaddafss_sc0_1w1sScanNet01-way 1-shotDownloaddafss_sc0_1w5sScanNet01-way 5-shotDownloaddafss_sc0_2w1sScanNet02-way 1-shotDownloaddafss_sc0_2w5sScanNet02-way 5-shotDownloadContactFor any questions or issues, feel free to reach out!Email: [Your Email Here]Join in our Communication Group (WeChat):<div style="text-align: left;"><img src="assets/wechat_qr.png" width="200"/></div>
