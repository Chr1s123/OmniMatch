# 04-1 向量基础设施选型与 OpenSearch 演进方向

## 1. 应用层能力要求

应用层需要同时满足以下能力：

- **Hybrid 加权融合**：在引擎层融合语义向量检索与全文检索，并支持运行时调权。
- **中文全文检索**：支持中文分词，并可使用 `analysis-ik` 一类分词插件。
- **标量过滤**：支持品类精确过滤、时间范围过滤等结构化约束。
- **开源协议清晰**：允许项目以可控、可演进的方式自托管。

候选方案中，OpenSearch、Elasticsearch 8+ 和 Weaviate 能原生覆盖主要 Hybrid 能力；
综合中文全文检索、标量过滤、线性融合与开源协议后，应用层选择 OpenSearch。

该选择并不意味着 OpenSearch 在每个单项上都最强，而是它能在同一个引擎内组合：

```text
语义召回 + 全文匹配 + 标量过滤 + 运行时线性加权
```

## 2. 最终双栈架构

| 层级 | 技术选型 | 负责场景 | 演进方向 |
| --- | --- | --- | --- |
| 商品召回层 | Faiss，HNSW + Inner Product | 三塔向量商品召回、`ItemSearch` | 数据与服务规模扩大后演进至 Milvus |
| 应用检索层 | OpenSearch，Hybrid Query + Cosine + HNSW | 长期记忆 Store、`CategoryInsight` 知识库 RAG | 保持 OpenSearch Hybrid 检索 |
| 统一 Embedding | 三塔 Query 塔，以 HTTP 服务暴露 | Faiss 离线灌库、OpenSearch 在线增量写入和查询 | 版本化发布并保持两栈向量空间一致 |

### 2.1 商品召回层：Faiss

Faiss 直接读取 NumPy 向量，适合千万级至亿级 SKU 的自训练向量检索。当前目标索引为
HNSW + Inner Product，并以单机 `P99 < 50ms` 作为性能验收目标。

商品召回层不使用 OpenSearch。该层的核心工作负载是纯向量 ANN；增加数据库服务会引入
网络、序列化和数据复制开销。生产阶段在容量、持久化、高可用或水平扩展需求出现后，
再将 Faiss 索引服务演进为 Milvus。

### 2.2 应用检索层：OpenSearch

OpenSearch 服务于两类非商品召回场景：

- 长期记忆 Store：保留黑名单等全量约束，并对偏好记忆执行向量 Top-K。
- `CategoryInsight` 知识库 RAG：组合语义、中文全文和标量过滤。

OpenSearch 不替代 Faiss 的商品 Top-100 召回，也不与 Faiss 竞争同一次 `item_search`
请求。两个引擎按工作负载分层，而不是把同一批商品候选做并行或串联检索。

### 2.3 共享 Query 塔

Faiss 与 OpenSearch 必须复用同一个版本化 Query 塔 Embedding 服务：

- 商品 Item 向量离线灌入 Faiss 时记录模型版本。
- 记忆和知识文档在线增量写入 OpenSearch 时记录模型版本。
- 查询两个引擎时使用与目标索引兼容的 Query 塔版本。
- 模型升级采用新索引构建、验证、别名切换和旧索引回收流程，禁止原地混写不同空间。

共享 Query 塔的目的是统一编码语义与版本治理，避免两栈之间出现向量空间漂移。

## 3. 向量检索与标量过滤的组合

常见组合方式有三种：

| 方式 | 行为 | 主要风险 |
| --- | --- | --- |
| Pre-filtering | 先按标量条件裁剪，再执行向量检索 | 候选过少时可能发生召回塌方 |
| Post-filtering | 先执行向量 Top-K，再按标量条件过滤 | Top-K 之外的相关结果会被提前丢弃 |
| Hybrid Fusion | 语义、全文与过滤独立计算后在引擎层融合 | 需要定义归一化和可观测的融合权重 |

应用层采用 Hybrid Fusion，使权重可以在运行时调整；商品召回层保持纯向量 ANN，避免把
应用层的全文和过滤职责耦合进三塔召回服务。

## 4. 工作负载路由

```text
ItemSearch
  -> User 塔 + Query 塔
  -> 请求向量融合
  -> Faiss Item 索引 Top-100
  -> Reranker Top-10
  -> ItemPicker

MemoryRead / MemoryWrite
  -> 共享 Query 塔 Embedding
  -> OpenSearch 记忆索引

CategoryInsight
  -> 共享 Query 塔 Embedding
  -> OpenSearch 知识库 Hybrid Query
```

## 5. 本章小结

1. 项目中的向量检索分布在商品召回、长期记忆和知识库 RAG 三类场景。
2. 商品召回使用 Faiss，生产规模扩大后可演进为 Milvus。
3. 长期记忆和知识库 RAG 使用 OpenSearch，以获得语义、全文、标量过滤和线性融合能力。
4. Faiss 与 OpenSearch 按工作负载分层，不共同处理同一次商品召回。
5. 两栈共享版本化 Query 塔 Embedding 服务，并通过索引版本治理避免向量空间漂移。
