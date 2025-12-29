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
