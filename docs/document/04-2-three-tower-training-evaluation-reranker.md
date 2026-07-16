# 04-2 三塔召回训练、评测与 Reranker

## 1. Embedding 质量的核心因素

Embedding 准确性主要由数据、Loss 设计和 Hard Negative 迭代决定，模型架构本身不是
唯一决定因素。负样本质量决定了召回模型的效果上限。

跨境商品场景必须处理假负样本：

- 同款商品的不同 SKU 不能被误标为负样本。
- 同一商品的不同语言描述不能被误标为负样本。
- 假负样本过滤同时使用商品同款识别和跨语言去重。

## 2. 三阶段训练范式

| 阶段 | 学习率 | 目标 |
| --- | --- | --- |
| CPT 领域持续预训练 | `1e-5` | 让基础模型适应电商领域及跨境商品语料 |
| SFT 监督对比精调 | `5e-6` | 使用 Curriculum Learning 学习 Query-Item 与用户行为信号 |
| DPO 偏好对齐 | `1e-6` | 对齐真实业务偏好与市场差异 |

三个阶段使用不同学习率，降低灾难性遗忘风险。

## 3. InfoNCE 改进

基础 InfoNCE 需要加入四项改进：

1. **动态温度**：跨语言样本使用 `0.02`，同语言样本使用 `0.05`。
2. **Hard Negative 加权**：Hard Negative 权重乘以 `2`。
3. **跨语言对齐辅助 Loss**：加入 `L_align`，约束多语言语义空间。
4. **假负样本过滤**：在构造训练批次前过滤同款 SKU 和多语言重复商品。

## 4. 评测要求

评测数据必须来自真实业务日志切分，不能只使用人工构造样本。

| 指标 | 验收阈值 |
| --- | --- |
| `Recall@100` | `>= 0.85` |
| `NDCG@10` | `>= 0.55` |
| 跨语言 Recall Gap | `<= 5%` |
| Cohen's Kappa | `>= 0.75` |

除离线质量指标外，还需要验证 Faiss 商品召回的单机延迟目标 `P99 < 50ms`。

## 5. Reranker

双塔和三塔本质上都是 bi-encoder，只能分别编码请求与商品，无法完整建模
`query x item` 的细粒度交叉信号。因此召回 Top-100 后必须增加 cross-encoder
Reranker，并输出 Top-10 给 ItemPicker。

### 5.1 训练流程

Reranker 采用主训练三阶段，再进行偏好对齐：

1. Pointwise BCE 热启动。
2. Pairwise MarginRanking 过渡。
3. Listwise ApproxNDCG，直接对齐排序指标。
4. 带市场标记的 DPO 偏好对齐。

### 5.2 Embedding 与 Reranker 协同

- 使用双向 MarginMSE 蒸馏，让 Embedding 与 Reranker 相互对齐。
- 使用渐进式 Hard Negative 挖掘迭代训练数据。
- 第二轮重点选择“Reranker 分数 `> 0.3` 但用户未购买”的样本，定位更困难的负样本。

### 5.3 模型与部署要求

- 基础模型：BGE-Reranker-v2-m3。
- 微调方式：LoRA。
- 延迟预算：`50-100ms`。
- 运行要求：GPU、FP16、批处理和启动 warmup。

## 6. 最终召回链路

```text
User / Query / Item 三塔编码
  -> 语义与个性化请求向量融合
  -> Faiss Top-100
  -> Cross-encoder Reranker Top-10
  -> ItemPicker
```

该链路把候选生成、精细相关性建模和 Agent 业务决策分成独立阶段。Faiss 保证大规模召回
效率，Reranker 补足 bi-encoder 的交叉建模上限，ItemPicker 再结合预算、物流、风险和
用户约束形成最终推荐。

## 7. 本章小结

1. 数据、Loss 和 Hard Negative 质量共同决定 Embedding 上限。
2. 三塔模型依次经过 CPT、SFT 和 DPO，并针对跨语言场景改进 InfoNCE。
3. 评测同时覆盖召回、排序、跨语言一致性和标注一致性。
4. Faiss Top-100 后使用 BGE-Reranker-v2-m3 生成 Top-10。
5. 最终由 ItemPicker 结合业务约束完成推荐决策。
