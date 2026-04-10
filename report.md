# VAD + DINOv3 Experiment Report

## Setup

| Config | Value |
|--------|-------|
| Model | VAD-Tiny + DINOv3 ViT-S/16 (frozen backbone) |
| Backbone | DINOv3 ViT-S/16, embed_dim=384, patch_size=16, register_tokens=4 |
| Config file | `projects/configs/VAD/VAD_tiny_dinov3_e2e.py` |
| GPUs | 1× (original paper: 8×) |
| samples_per_gpu | 16 (effective batch=16; original: 8GPU×1=8) |
| lr | 4e-4 (linear scaling: 16/8 × 2e-4) |
| Epochs | 24 (CosineAnnealing, warmup 500 iters) |
| Dataset | nuScenes |

---

## Results (Epoch 1 / 24)

> Note: epoch 1 only, for pipeline validation. Final results expected after full 24-epoch training.

### 3D Object Detection

| Metric | Value |
|--------|-------|
| mAP | 0.1592 |
| NDS | 0.2124 |
| mATE | 0.9380 |
| mASE | 0.3521 |
| mAOE | 0.9727 |
| mAVE | 1.0578 |
| mAAE | 0.4087 |

Per-class AP:

| Class | AP | ATE | ASE | AOE |
|-------|----|-----|-----|-----|
| car | 0.403 | 0.611 | 0.198 | 0.360 |
| truck | 0.125 | 0.910 | 0.290 | 0.721 |
| bus | 0.100 | 1.150 | 0.336 | 0.608 |
| trailer | 0.007 | 1.113 | 0.367 | 0.873 |
| construction_vehicle | 0.023 | 1.420 | 0.548 | 1.569 |
| pedestrian | 0.250 | 0.790 | 0.310 | 1.654 |
| motorcycle | 0.128 | 0.799 | 0.394 | 1.309 |
| bicycle | 0.105 | 0.782 | 0.339 | 1.430 |
| traffic_cone | 0.209 | 0.863 | 0.399 | - |
| barrier | 0.241 | 0.941 | 0.339 | 0.230 |

### Map Prediction (Chamfer Distance)

| Class | AP@0.5 | AP@1.0 | AP@1.5 | AP |
|-------|--------|--------|--------|----|
| divider | 0.029 | 0.151 | 0.271 | 0.151 |
| ped_crossing | 0.000 | 0.045 | 0.141 | 0.062 |
| boundary | 0.040 | 0.236 | 0.452 | 0.243 |
| **mAP** | | | | **0.152** |

### Motion Prediction

| Metric | Car | Pedestrian |
|--------|-----|------------|
| EPA | 0.2061 | 0.0308 |
| ADE | 1.0722 | 0.9895 |
| FDE | 1.5613 | 1.5285 |
| MR | 0.1636 | 0.2619 |

### Planning

| Metric | 1s | 2s | 3s |
|--------|----|----|----|
| L2 (m) | 0.544 | 0.931 | 1.386 |
| Obj Col (%) | 0.000 | 0.005 | 0.003 |
| Obj Box Col (%) | 0.518 | 0.811 | 1.227 |

---

## TODO

- [ ] Comparison with VAD-Tiny (ResNet backbone) baseline
