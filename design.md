# VAD + DINOv3 Backbone 设计文档

## 背景

本项目在 VAD (Vector-based Autonomous Driving, CVPR 2023) 基础上，将原有 ResNet-50 图像 backbone 替换为 Meta Research 的 DINOv3 视觉基础模型，探索更强视觉特征对端到端自动驾驶规划任务的影响。

**参考论文**:
- VAD: https://arxiv.org/abs/2303.12077
- DINOv3: https://github.com/facebookresearch/dinov3

---

## 架构对比

| 组件 | 原始 VAD (tiny) | VAD + DINOv3 |
|------|----------------|--------------|
| Backbone | ResNet-50 (25M) | DINOv3 ViT-S/16 (21M) |
| 预训练 | ImageNet supervised | LVD-1689M self-supervised |
| Backbone 输出 | 单尺度 (2048-d, stride=32) | 单尺度 (384-d, stride=16) |
| Neck | FPN (2048→256) | FPN (384→256) |
| BEV 编码器 | BEVFormer (3层) | BEVFormer (3层，不变) |
| BEV 分辨率 | 100×100 | 100×100 (不变) |
| 特征图大小 | ~20×11 (0.4x scale) | ~40×23 (0.4x scale, stride=16) |
| GridMask | 开启 | 关闭 (不适合 ViT) |
| Backbone 冻结 | 否 (lr×0.1) | 是 (frozen) |

### DINOv3 关键特性

- **架构**: Vision Transformer，patch_size=16
- **预训练**: 自监督 DINO loss，1.7B 张图片 (LVD-1689M)
- **输出**: patch token 特征，reshape 为 (B, 384, H/16, W/16)
- **位置编码**: 支持任意分辨率插值
- **模型变体**: ViT-S(384d), ViT-B(768d), ViT-L(1024d), ViT-H+(1280d), ViT-7B(1408d)

---

## 实现细节

### 新增文件

```
projects/mmdet3d_plugin/models/backbones/dinov3.py   # DINOv3 backbone wrapper
projects/configs/VAD/VAD_tiny_dinov3_e2e.py          # 训练/测试配置
design.md                                             # 本文档
```

### 修改文件

```
projects/mmdet3d_plugin/models/backbones/__init__.py  # 注册 DINOv3Backbone
projects/mmdet3d_plugin/__init__.py                   # 导入 DINOv3Backbone
```

### DINOv3Backbone 接口

```python
img_backbone=dict(
    type='DINOv3Backbone',
    model_name='dinov3_vits16',           # 模型名称
    repo_path='third_party/dinov3',        # DINOv3 仓库本地路径
    pretrained_weights='ckpts/dinov3_vits16.pth',  # checkpoint 路径
    frozen=True,                           # 冻结 backbone 参数
)
```

Forward 流程:
1. 输入: `(B×N_cam, 3, H, W)` → DINOv3 patch embedding
2. 提取最后一层 patch tokens via `get_intermediate_layers(n=1, reshape=True)`
3. 输出: `((B×N_cam, 384, H/16, W/16),)` - tuple 格式兼容 FPN

---

## 服务器环境准备

### 1. 克隆 DINOv3 仓库

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
```

### 2. 下载 DINOv3 权重

从 HuggingFace 下载 ViT-S/16 checkpoint:

```bash
mkdir -p ckpts
# 方式1: huggingface-cli
huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m \
    --local-dir ckpts/dinov3-vits16 --repo-type model

# 方式2: python
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/dinov3-vits16-pretrain-lvd1689m',
    filename='pytorch_model.bin',
    local_dir='ckpts/'
)
"
# 重命名为配置文件期望的名称
mv ckpts/pytorch_model.bin ckpts/dinov3_vits16.pth
```

> 注意: HuggingFace 上的 DINOv3 checkpoint 格式可能为 safetensors 或 transformers 格式。
> 若权重 key 格式不匹配，需要转换。详见下方"权重转换"章节。

### 3. 权重格式转换（如需要）

DINOv3 通过 torch.hub.load 加载的模型期望 `model.load_state_dict(state_dict)` 兼容的格式。若 HF 权重包含额外前缀，可用以下脚本转换：

```python
import torch
ckpt = torch.load('ckpts/dinov3_vits16.pth', map_location='cpu')
# 若有嵌套结构
if 'model' in ckpt:
    ckpt = ckpt['model']
# 检查 key 是否有前缀 (如 'backbone.')
keys = list(ckpt.keys())
print(keys[:5])
# 如有前缀, 去掉:
# ckpt = {k.replace('backbone.', ''): v for k, v in ckpt.items()}
torch.save(ckpt, 'ckpts/dinov3_vits16_clean.pth')
```

### 4. 验证数据集路径

确认 `data/nuscenes/` 下有:
- `vad_nuscenes_infos_temporal_train.pkl`
- `vad_nuscenes_infos_temporal_val.pkl`
- `nuscenes_map_anns_val.json`

---

## 训练命令

### 单 GPU 快速验证（推荐先跑通）

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    projects/configs/VAD/VAD_tiny_dinov3_e2e.py \
    --work-dir work_dirs/vad_tiny_dinov3 \
    --launcher none
```

### 多 GPU 训练

```bash
bash tools/dist_train.sh \
    projects/configs/VAD/VAD_tiny_dinov3_e2e.py \
    <NUM_GPUS> \
    --work-dir work_dirs/vad_tiny_dinov3
```

---

## 测试命令

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    projects/configs/VAD/VAD_tiny_dinov3_e2e.py \
    work_dirs/vad_tiny_dinov3/epoch_24.pth \
    --launcher none \
    --eval bbox \
    --tmpdir tmp
```

---

## 预期结果对比

以下为参考指标（VAD 论文 Table 1，nuScenes val set）：

| 模型 | L2 (m) ↓ | Col. Rate (%) ↓ |
|------|-----------|-----------------|
| VAD-Tiny (论文) | 0.54 | 0.04 |
| VAD-Base (论文) | 0.17 | 0.07 |
| VADv2 (论文) | — | — |
| **VAD-Tiny + DINOv3** | TBD | TBD |

> 由于本次实验为验证性跑通（24 epochs，冻结 backbone），不追求完全复现论文数值。
> 关注指标变化趋势和收敛稳定性。

---

## 理解与分析（待填写训练结果后完善）

### 为什么用 DINOv3？

1. **更丰富的语义特征**: DINOv3 在 17 亿张图片上自监督预训练，特征泛化性强，对场景理解（车辆、行人、路面语义）比 ImageNet-supervised ResNet 更具优势。

2. **密集特征质量**: DINO 系列以高质量 dense 特征著称，特别适合需要空间对应的任务（BEV 投影需要精确的视觉-空间对应关系）。

3. **无需标注数据**: 自监督预训练不依赖驾驶数据标注，具有更好的迁移性。

### 主要挑战

1. **单尺度输出**: ViT 天然输出单尺度特征，而 ResNet 的多尺度特征对检测任务有优势。本实验基于 VAD_tiny（单尺度配置），规避了此问题。

2. **stride=16 vs stride=32**: DINOv3 patch_size=16 产生更高分辨率特征图（stride=16），比 ResNet 的 stride=32 细节更丰富，但显存开销略高。

3. **位置编码插值**: DINOv3 预训练于固定分辨率，推理时对非标准分辨率（如 640x368）进行 bicubic 插值，可能有轻微精度损失。

4. **框架版本差异**: VAD 基于 mmdet 2.14 + PyTorch 1.9，而 DINOv3 开发于更新的 PyTorch 版本。通过 torch.hub.load(source='local') 方式加载可以避免 transformers 库版本依赖。
