# AgentLoop、三塔召回与双栈检索架构设计

## 1. 背景

OmniMatch 已经从固定工具链演进为观察驱动的购物 Agent，但当前实现仍主要依赖外部商品
Provider 和占位模块。下一阶段需要统一以下目标架构：

- 主智能体使用观察驱动的 AgentLoop。
- 多 Agent 协同只使用同质子 AgentLoop fork。
- 商品召回使用 User / Query / Item 三塔以及语义、个性化双通道。
- 商品 ANN 使用 Faiss，生产规模扩大后保留演进至 Milvus 的接口边界。
- 长期记忆和 `CategoryInsight` RAG 使用 OpenSearch。
- Faiss 与 OpenSearch 共享同一个版本化 Query 塔服务，避免查询侧向量空间漂移。
- Faiss Top-100 后使用 cross-encoder Reranker 生成 Top-10，再交给 ItemPicker。

本设计是覆盖多个子系统的总架构 spec。实施工作拆成四份独立 plan，每份都必须产生可单独
测试和验收的软件增量。

参考资料：

- [三塔向量召回与语义个性化](../../document/04-0-llm-three-tower-vector-recall.md)
- [向量基础设施选型与 OpenSearch](../../document/04-1-vector-infrastructure-and-opensearch.md)
- [训练、评测与 Reranker](../../document/04-2-three-tower-training-evaluation-reranker.md)

## 2. 当前实现状态

本节记录 2026-07-16 的仓库事实。后续章节描述目标架构，不代表这些能力已经实现。

| 组件 | 状态 | 当前实现 | 目标差距 |
| --- | --- | --- | --- |
| 主 `CompetitionAgentLoop` | 部分实现 | 已支持 LLM Action、观察驱动循环、终止 Action 和 `max_steps` | 缺少统一时间、并发、子 Agent 和资源预算 |
| `app/agent/dispatch_tool.py` | 原型且未接入 | 使用 `asyncio.gather` 并行调用平台搜索与物流函数 | 不是 AgentLoop fork，未被主循环或 Tool Registry 调用，缺少隔离、预算、取消与合并协议 |
| 同质子 AgentLoop fork | 未实现 | 只有 `subagent_started` 和 `subagent_finished` 原型事件 | 缺少 fork Action、子循环、结果协议和生命周期控制 |
| User / Query / Item 三塔 | Mock stub | 各编码函数只返回与输入长度相关的一维数组 | 缺少真实模型接口、批处理、归一化、版本和模型服务 |
| 语义与个性化融合 | 未实现 | 无融合代码 | 缺少投影、权重、冷启动和兼容性校验 |
| ANN | Mock stub | `search_ann()` 生成虚假 ID 和分数 | 缺少 Faiss、真实 Item 索引和索引生命周期 |
| `ItemSearch` | 部分实现 | 直接调用 Product Provider | 尚未接入三塔、Faiss 和 Reranker |
| Reranker | 未实现 | 无接口、模型或推理服务 | 缺少 Top-100 到 Top-10 的精排链路 |
| Memory | Mock stub | 只有进程内字符串列表 | 未接入 AgentLoop 或 OpenSearch |
| `CategoryInsight` | 部分实现 | 调用 WebSearch Provider 并拼接固定属性 | 尚未接入 OpenSearch Hybrid RAG |
| OpenSearch | 未实现 | 无客户端、索引或查询代码 | 缺少记忆与知识库索引、Hybrid Query 和健康检查 |
| Milvus | 后续演进 | 无实现 | 当前阶段只保留 ANN Provider 兼容边界 |
| 三塔与 Reranker 训练 | 未实现 | `app/eval` 只评估 Agent 输出 | 缺少数据、训练、召回评测、模型发布与索引切换流水线 |

## 3. 目标与非目标

### 3.1 目标

- 保持主 AgentLoop 对检索基础设施无感，只通过稳定 Tool 契约使用能力。
- 将现有平台并行函数原型替换为真正的同质子 AgentLoop fork。
- 用三塔双通道请求向量在 Faiss Item 索引中召回 Top-100。
- 用 Reranker 将 Top-100 精排为 Top-10，再由 ItemPicker 处理业务约束。
- 用 OpenSearch 实现长期记忆和 `CategoryInsight` Hybrid RAG。
- 为模型、向量和索引建立显式版本兼容检查。
- 保持 `test` 和 `submission` 无密钥、无网络、确定性运行。
- 建立从训练数据到模型 Bundle、Faiss 索引和灰度切换的可审计流程。

### 3.2 非目标

- 当前阶段不实施 Milvus 集群，只定义可替换的 ANN Provider 契约。
- 不使用 OpenSearch 替代 Faiss 的商品 ANN，也不让两个引擎共同处理同一次商品召回。
- 不引入异质专家 Agent、角色型 Agent 团队或其他 Agent 框架。
- 子 Agent 不直接生成面向用户的最终答案。
- 不声称在缺少真实业务日志和 GPU 的情况下已经达到生产模型指标。
- 不在 React 前端复制召回、融合、精排或 Agent 决策逻辑。

## 4. 总体架构

```text
主 CompetitionAgentLoop
  |-- 同质子 CompetitionAgentLoop fork
  `-- ToolRegistry
       |-- ItemSearch
       |    |-- UserTower + QueryTower
       |    |-- 语义/个性化请求向量融合
       |    |-- Faiss Item Index Top-100
       |    |-- Cross-encoder Reranker Top-10
       |    `-- ItemPicker
       |-- MemoryRead / MemoryWrite
       |    `-- OpenSearch Memory Index
       `-- CategoryInsight
            `-- OpenSearch Knowledge Hybrid Query
```

层级职责：

- **Agent 层**决定是否调用工具、是否 fork、何时澄清、失败或完成。
- **Tool 层**拥有业务契约，隐藏三塔、ANN、OpenSearch 和模型服务细节。
- **Recall 层**负责商品候选生成及精排，不处理最终推荐文案。
- **Application Retrieval 层**负责记忆和知识检索，不承担商品 Item ANN。
- **Training 层**独立训练三塔与 Reranker，通过版本化模型 Bundle 和索引交付运行时。

## 5. 主 AgentLoop 与同质子 AgentLoop fork

### 5.1 主循环

主 Agent 使用现有观察驱动范式：

```text
Think -> Act -> Observe -> Reflect
```

循环由结构化 Action 驱动，并在 `finish`、`clarify`、`fail`、步数耗尽、时间耗尽或取消时
终止。只有主 Agent 可以产生最终用户回答。

### 5.2 Fork Action

Action Schema 新增 `fork`。`ForkRequest` 至少包含：

- `task_id`：父任务内稳定且唯一的子任务标识。
- `objective`：子 Agent 的单一、可验收目标。
- `allowed_tools`：子 Agent 可调用的 Tool 白名单。
- `context_snapshot`：从父 Agent 复制的只读最小上下文。
- `max_steps`：子循环步数预算。
- `timeout_seconds`：子循环时间预算。
- `merge_key`：父 Agent 合并结构化结果时使用的类别键。

子 Agent 仍运行 `CompetitionAgentLoop`，复用相同的 LLM Provider、Action Schema、
Tool Registry 实现和事件协议。它使用独立 Tool Registry、Observation History 和输出状态，
不能修改父 Agent 的可变对象。

第一阶段默认限制：

- `max_fork_depth=1`
- `max_parallel_subagents=4`
- `subagent_max_steps=4`
- `subagent_timeout_seconds=30`

这些限制由类型化配置覆盖。达到深度、并发或父任务预算后，fork 请求必须被拒绝并形成
可观察的结构化结果，不能静默启动额外任务。

### 5.3 子 Agent 结果与合并

`SubAgentResult` 包含：

- `task_id`
- `status`：`completed`、`failed`、`cancelled` 或 `timed_out`
- `result`
- `observations`
- `warnings`
- `error`
- `step_count`
- `elapsed_ms`

父 Agent 按 `task_id` 稳定排序后合并结果，避免异步完成顺序影响提示词和测试。单个子 Agent
失败时保留成功结果和失败证据；全部子 Agent 失败时，主 Agent 才进入 `clarify` 或 `fail`。
父任务被取消、超时或耗尽预算时，所有未完成子任务必须被取消。

现有 `dispatch_tool.py` 不能继续被描述为“子 Agent 实现”。实施时应将其替换为 fork
调度器，或删除其中已被新调度器覆盖的未接入代码。

## 6. 三塔与双通道召回

### 6.1 塔职责

| 塔 | 输入 | 输出职责 |
| --- | --- | --- |
| User | 用户画像、点击、收藏、购买等历史行为 | 长期偏好向量 |
| Query | 当前查询和 Query RAG 增强信息 | 当前语义意图向量 |
| Item | 商品标题、类目、属性和历史成交 Query | Faiss Item 索引向量 |

模型 Bundle 共同发布 User、Query、Item 编码器和个性化投影层，保证输出维度、归一化与
向量空间兼容。

### 6.2 融合公式

请求侧使用以下固定语义：

```text
q_sem = L2_normalize(QueryTower(query_features))
u_pref = L2_normalize(UserTower(user_features))

q_personal = L2_normalize(
    PersonalizationProjection(concat(q_sem, u_pref))
)

request_vector = L2_normalize(
    alpha * q_sem + beta * q_personal
)

item_vector = L2_normalize(ItemTower(item_features))

results = Faiss_IP_search(
    request_vector,
    item_vector_index,
    top_k=100
)
```

约束：

- `alpha` 和 `beta` 必须在 `[0, 1]` 范围内且总和为 `1`。
- 第一版基线为 `alpha=0.7`、`beta=0.3`，允许通过配置覆盖。
- 无有效用户历史时强制使用 `alpha=1`、`beta=0`。
- 语义与个性化向量必须具有相同维度、归一化方式和可比较尺度。
- Embedding 维度来自模型 manifest，不在业务代码中写死。
- Faiss 使用 L2 归一化向量和 Inner Product，使分数等价于 Cosine 相似度。

### 6.3 Faiss 索引

第一阶段使用 Faiss HNSW + Inner Product，并返回 `top_k=100`。索引 manifest 至少包含：

- `model_bundle_version`
- `embedding_dimension`
- `normalization`
- `distance_metric`
- `item_count`
- `created_at`
- `checksum`

请求模型与索引的 Bundle 版本、维度、归一化或距离度量不兼容时，必须拒绝检索。禁止在
同一个索引中混写不同模型空间。规模、持久化、高可用或水平扩展需求超过单机边界后，
ANN Provider 可以演进至 Milvus，但 Tool 和 Agent 契约保持不变。

### 6.4 Reranker 与 ItemPicker

Faiss Top-100 进入 cross-encoder Reranker。Reranker 同时读取 Query 与 Item 文本及必要的
市场标记，输出 Top-10 和精排分数。ItemPicker 再结合预算、物流、黑名单、硬约束、风险、
证据质量和推荐理由生成最终候选。

Reranker 不取代 ItemPicker：前者负责 Query-Item 交叉相关性，后者负责可解释业务决策。

## 7. Faiss 与 OpenSearch 双栈

### 7.1 固定职责边界

| 引擎 | 场景 | 不负责 |
| --- | --- | --- |
| Faiss | `ItemSearch` 商品 Top-100 ANN | 长期记忆、知识 RAG、全文检索 |
| OpenSearch Memory | 黑名单全量约束、偏好记忆向量 Top-K | 商品 Item ANN |
| OpenSearch Knowledge | `CategoryInsight` 语义、全文、标量过滤 Hybrid Query | 商品 Item ANN |

OpenSearch 不参与 Faiss 商品候选的并行召回、串联过滤或二次 ANN。两个引擎按工作负载
分层，而不是共同服务同一次商品检索。

### 7.2 共享 Query 塔

Faiss 请求侧和 OpenSearch 查询侧复用同一个版本化 Query 塔 HTTP 服务。Faiss 商品向量
仍由同一模型 Bundle 中的 Item 塔生成，不能用 Query 塔替代 Item 塔。

按参考资料要求，OpenSearch 记忆和知识文档的向量也由同一 Query 塔服务编码。写入文档
和发起查询时都记录模型版本；索引别名只能指向与当前 Query 塔兼容的索引。

模型升级采用：

```text
发布新模型 Bundle
  -> 构建新 Faiss / OpenSearch 索引
  -> 离线质量与兼容性验证
  -> 灰度查询
  -> 原子切换索引别名或当前索引指针
  -> 保留回滚窗口
  -> 回收旧索引
```

### 7.3 OpenSearch Hybrid Query

应用层组合向量分数、BM25 和标量过滤：

```text
hybrid_score =
    vector_weight * normalized_vector_score
    + text_weight * normalized_bm25_score
```

第一版基线为 `vector_weight=0.6`、`text_weight=0.4`。两者在 `[0, 1]` 范围内且总和为
`1`，允许运行时调权。中文字段使用可配置中文分词器；品类、语言、市场、时间和记忆类型
使用结构化字段过滤。

## 8. 训练、评测与模型发布

### 8.1 数据契约

训练日志的最小字段包括：

- `user_id`
- `query`
- `item_id`
- `market`
- `language`
- `event_type`
- `impression_position`
- `ranker_score`
- `timestamp`
- `item_group_id`

`item_group_id` 用于同款不同 SKU 去重。跨语言假负样本过滤还需要规范化商品属性和语义
相似度，避免同一商品的不同语言描述互为负样本。数据集按时间切分，禁止同一用户事件或
商品版本跨训练、验证和测试集泄漏。

### 8.2 三塔训练

```text
日志清洗与时间切分
  -> 同款 SKU / 跨语言假负样本过滤
  -> CPT，lr=1e-5
  -> SFT + Curriculum Learning，lr=5e-6
  -> DPO，lr=1e-6
  -> 离线评测
  -> 模型 Bundle
```

InfoNCE 要求：

- 跨语言样本温度 `0.02`，同语言样本温度 `0.05`。
- Hard Negative 权重为普通负样本的 `2` 倍。
- 加入跨语言对齐辅助损失 `L_align`。
- 在训练批次生成前过滤假负样本。
- 成交、点击、曝光和精排一致性使用可配置多任务 Loss；实际权重必须写入训练配置与报告。

### 8.3 Reranker 训练与推理

Reranker 采用以下顺序：

```text
Pointwise BCE
  -> Pairwise MarginRanking
  -> Listwise ApproxNDCG
  -> 带市场标记的 DPO
```

基础模型为 BGE-Reranker-v2-m3，使用 LoRA 微调。Embedding 与 Reranker 使用双向
MarginMSE 蒸馏和渐进式 Hard Negative 挖掘；第二轮重点挖掘 Reranker 分数 `> 0.3`
但用户未购买的样本。

运行时使用 GPU、FP16、批处理和 warmup，Top-100 到 Top-10 的延迟预算为 `50-100ms`。

### 8.4 发布门槛

| 指标 | 门槛 |
| --- | --- |
| `Recall@100` | `>= 0.85` |
| `NDCG@10` | `>= 0.55` |
| 跨语言 Recall Gap | `<= 5%` |
| Cohen's Kappa | `>= 0.75` |
| Faiss 单机召回 P99 | `< 50ms` |
| Reranker 推理 P99 | `<= 100ms` |

未满足门槛的模型只能保留为实验产物，不能切换生产索引。仓库当前没有真实业务日志和
GPU，因此实施计划只能交付可运行流水线、fixture、报告格式和发布门禁，不能把上述指标
标记为已经达成。

## 9. 配置与运行 Profile

新增设置应继续集中在 `OmniMatchSettings`，至少覆盖：

- Agent fork 深度、并发、步数和超时。
- Embedding Provider、模型 Bundle 版本和服务地址。
- `alpha`、`beta`、向量维度兼容策略。
- ANN Provider、Faiss 索引路径、Top-K 和查询参数。
- Reranker Provider、模型版本、批大小、Top-K 和超时。
- OpenSearch 地址、索引别名、认证、Hybrid 权重和超时。
- 各降级路径是否允许启用。

Profile 约束：

- `test` 使用确定性 fake 编码器、内存 ANN、fake Reranker 和 fake OpenSearch，不调用网络。
- `submission` 无需密钥或外部服务，使用确定性 fixture 走相同接口并披露 placeholder 模式。
- `dev` 使用显式配置的真实服务，缺少必要地址、模型或索引时快速失败。

## 10. 错误处理与降级

| 故障 | 行为 |
| --- | --- |
| Query/User 编码失败 | 停止该次向量召回，记录 Provider 错误 |
| 模型与 Faiss 索引不兼容 | 拒绝检索，不允许忽略版本或维度错误 |
| Faiss 不可用 | 仅在配置允许时回退当前 Product Provider，并标记 `recall_mode="provider_fallback"` |
| Reranker 失败 | 允许按 Faiss 分数截取 Top-10，披露未执行精排 |
| Category RAG 失败 | 回退当前 WebSearch Provider并记录证据来源 |
| 偏好记忆读取失败 | 允许无个性化继续并警告 |
| 黑名单或硬约束读取失败 | 不得静默忽略，任务进入 `clarify` 或 `fail` |
| 单个子 Agent 失败 | 合并成功结果并保留失败信息 |
| 全部子 Agent 失败 | 主 Agent 进入 `clarify` 或 `fail` |
| 父任务取消、超时或耗尽预算 | 取消全部未完成子任务 |

降级不能改变证据真实性。最终结果必须披露所用召回模式、Provider 模式、缺失能力和不确定性。

## 11. 可观测性与安全

事件与 Trace 需要增加：

- Agent：fork 深度、子任务 ID、步数、超时、取消、耗时和状态。
- Embedding：模型 Bundle 版本、维度、编码耗时和缓存命中。
- Faiss：索引版本、候选数、ANN 耗时、`alpha`、`beta` 和召回模式。
- Reranker：模型版本、输入数、输出数、批次数和耗时。
- OpenSearch：索引别名、Hybrid 权重、过滤摘要、命中数和耗时。

Trace 禁止记录 API Key、认证头、原始完整用户历史和完整敏感画像。用户 ID 必须使用适合
分析的匿名标识；Provider 原始响应继续只保存脱敏摘要。

## 12. 测试策略

### 12.1 单元与契约测试

- 向量 L2 归一化、融合公式、权重校验和冷启动退化。
- 模型 Bundle 与索引 manifest 兼容性。
- Fork 预算、深度、稳定合并、取消和部分失败。
- Embedding、ANN、Reranker、Memory 和 Knowledge Provider 契约。
- OpenSearch Hybrid 权重和标量过滤请求构造。

### 12.2 集成与端到端测试

- 使用小型确定性 Item fixture 构建真实 Faiss 索引并验证 Top-K。
- 使用 Docker OpenSearch 验证记忆索引、知识索引、中文字段和 Hybrid Query。
- 验证 `ItemSearch -> Faiss Top-100 -> Reranker Top-10 -> ItemPicker`。
- 验证主 Agent fork、子任务部分失败和父任务取消。
- 验证 OpenSearch 故障时 Category RAG 降级，以及硬约束读取失败时拒绝静默推荐。
- 现有后端测试、submission smoke 和前端 build 必须继续通过。

### 12.3 离线评测与性能测试

- 真实日志时间切分上的 Recall、NDCG、跨语言差距和一致性指标。
- Faiss 与 Reranker 的 P50、P95、P99。
- 性能测试与普通 pytest 分离，输出机器信息、模型版本、索引版本和完整参数。

## 13. 实施拆分

本设计拆成四份 plan，按以下顺序实施：

1. **同质 AgentLoop fork**：替换 `dispatch_tool.py` 原型，增加 fork Action、隔离、预算、
   合并、取消、事件和测试。
2. **三塔 Faiss 召回与 Reranker 推理**：替换 `app/recall` mock，接入向量融合、Faiss、
   manifest、Reranker 和 ItemSearch。
3. **OpenSearch 应用检索**：实现长期记忆、硬约束读取、CategoryInsight Hybrid RAG、
   共享 Query 塔和降级路径。
4. **训练、评测与模型生命周期**：实现数据契约、样本构造、三阶段训练、Reranker 训练、
   离线评测、模型 Bundle、索引构建和发布门禁。

每份 plan 必须在开头重复与该阶段有关的当前状态和明确非目标，避免将本总 spec 中的未来
能力误认为已有实现。

## 14. 验收标准

- 总 spec 明确区分已实现、部分实现、未实现和后续演进。
- `dispatch_tool.py` 不再被描述为真正的子 AgentLoop。
- 主 Agent 能 fork 同质子 AgentLoop，并对隔离、预算、取消和合并进行测试。
- 三塔输出、双通道融合和 Faiss Item 索引具有版本化契约。
- 商品召回严格执行 Faiss Top-100、Reranker Top-10、ItemPicker 的职责链。
- OpenSearch 仅服务长期记忆和 CategoryInsight RAG，不进入商品 ANN。
- Faiss 与 OpenSearch 查询侧共享版本化 Query 塔服务，Faiss Item 向量由 Item 塔生成。
- 无用户历史时退化为纯语义请求向量。
- 模型或索引不兼容时拒绝检索。
- 故障降级可配置、可观察并在最终结果中披露。
- `test` 与 `submission` 保持确定性、无外部网络依赖。
- 训练和发布流水线输出可审计配置、报告与 Bundle，但不虚构未实测的生产指标。
