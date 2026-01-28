# 多模态情感分类系统 (Multimodal Sentiment Classification)

基于文本和图像的多模态情感分类系统，将社交媒体内容分类为 **正向(positive)**、**中立(neutral)**、**负向(negative)** 三种情感。

## 📊 实验结果

### 消融实验结果

| 模态 | 验证准确率 | 验证 Macro-F1 |
|------|-----------|--------------|
| Text-only | 68.25% | 0.543 |
| Image-only | 65.00% | 0.533 |
| **Multimodal** | **72.75%** | **0.591** |

### 结论
- 多模态融合相比最好的单模态（文本）提升 **+4.5%**
- 门控融合机制有效整合文本和图像信息

---

## 🔧 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖版本

| 库 | 版本要求 | 用途 |
|---|---------|------|
| Python | >= 3.8 | 编程语言 |
| PyTorch | >= 1.10 | 深度学习框架 |
| torchvision | >= 0.11 | 图像处理 |
| transformers | >= 4.20 | BERT模型 |
| scikit-learn | >= 1.0 | 数据划分与评估 |
| Pillow | >= 8.0 | 图像读取 |
| matplotlib | >= 3.5 | 可视化 |
| tqdm | >= 4.60 | 进度条 |
| numpy | >= 1.20 | 数值计算 |

### GPU支持（可选但推荐）
- CUDA >= 11.0
- cuDNN >= 8.0

---

## 📁 代码文件结构

```
ai5/
├── train.py          # 主训练脚本（包含模型定义、训练、评估）
├── requirements.txt         # Python依赖列表
├── README.md               # 项目说明文档
│
├── project5/               # 数据目录
│   ├── train.txt           # 训练集标签 (guid,tag)
│   ├── test_without_label.txt  # 测试集 (guid,tag=null)
│   └── data/               # 原始数据文件
│       ├── {guid}.txt      # 文本文件
│       └── {guid}.jpg      # 图像文件
│
└── output/             # 输出目录
    └── {timestamp}/        # 按时间戳组织的实验结果
        └── run_*/          # 单次运行结果
            ├── args.json           # 超参数配置
            ├── best_model.pt       # 最优模型权重
            ├── epoch_log.csv       # 每轮训练指标
            ├── summary.json        # 最终结果摘要
            ├── test_predictions.csv # 测试集预测结果
            ├── training_curves.png  # 训练曲线图
            └── confusion_matrix.png # 混淆矩阵
```

---

## 🚀 执行代码的完整流程

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/ai5.git
cd ai5
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据

确保 `project5/` 目录结构如下：
```
project5/
├── train.txt              # 格式: guid,tag (如: 1,positive)
├── test_without_label.txt # 格式: guid,tag (tag为null)
└── data/
    ├── 1.txt, 1.jpg       # 样本1的文本和图像
    ├── 2.txt, 2.jpg       # 样本2的文本和图像
    └── ...
```

### 4. 运行训练

#### 单次融合实验
```bash
python train.py \
    --seed 42 \
    --fusion gated \
    --modality multimodal \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --lr_head_mult 5.0 \
    --text_model bert-base-uncased \
    --text_clean none \
    --patience 5 \
    --use_amp
```

#### 消融实验（同时运行 text/image/multimodal）
```bash
python train.py \
    --run_ablation \
    --seed 42 \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --lr_head_mult 5.0 \
    --text_model bert-base-uncased \
    --text_clean none \
    --patience 5 \
    --use_amp
```

### 5. 查看结果

训练完成后，结果保存在 `output/{timestamp}/` 目录：

```bash
# 查看实验结果摘要
cat output/最新时间戳/run_multimodal_gated/summary.json

# 查看测试集预测
cat output/最新时间戳/run_multimodal_gated/test_predictions.csv
```

---

## ⚙️ 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--modality` | multimodal | 模态选择: text/image/multimodal |
| `--fusion` | gated | 融合方式: late/gated/attention |
| `--text_model` | bert-base-uncased | 文本编码器 |
| `--image_backbone` | resnet18 | 图像编码器 |
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--lr` | 2e-5 | 基础学习率 |
| `--lr_head_mult` | 5.0 | 分类头学习率倍数 |
| `--dropout` | 0.2 | Dropout率 |
| `--patience` | 5 | 早停耐心值 |
| `--use_amp` | False | 启用混合精度训练 |
| `--text_clean` | basic | 文本清洗模式: basic/none |
| `--run_ablation` | False | 运行消融实验 |

---

## 🏗️ 模型架构

```
输入
 ├── 文本 ──→ BERT-base-uncased ──→ [CLS] 768维 ──→ 投影层 256维 ──┐
 │                                                                  │
 └── 图像 ──→ ResNet18 ──────────→ 特征 512维 ──→ 投影层 256维 ──┤
                                                                    │
                                                      门控融合 (Gated Fusion)
                                                                    │
                                                                    ▼
                                                          分类器 512→256→3
                                                                    │
                                                                    ▼
                                                    positive / neutral / negative
```

### 门控融合机制
```python
gate = sigmoid(W @ concat(text_feat, image_feat))
text_weighted = text_feat * gate
image_weighted = image_feat * (1 - gate)
output = concat(text_weighted, image_weighted)
```

---

## 📈 训练输出示例

```
run_multimodal_gated Epoch 1/10: 100%|████| 225/225 [00:43<00:00, loss=0.946]
[run_multimodal_gated] Epoch 1: loss=0.9465 val_acc=0.6450 val_f1=0.4050

run_multimodal_gated Epoch 2/10: 100%|████| 225/225 [00:43<00:00, loss=0.780]
[run_multimodal_gated] Epoch 2: loss=0.7802 val_acc=0.7000 val_f1=0.5128

...

训练完成，输出目录：output/20260128_114439
```

---
