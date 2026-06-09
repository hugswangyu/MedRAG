# RAG 医疗问答改进与评估计划

## Summary
把系统升级为“查询规范化 + 智能路由 + 按意图回答 + 个人病例库 + RAGAS 评估”的闭环：

- 查询先做中文医学规范化，再进入 KG、公共向量库、个人病例库路由。
- 回答 prompt 改为按意图模板生成，不再强制四/五段式。
- 用户病例按用户隔离保存、摘要、切块、检索，只服务当前用户。
- 最后用 RAGAS 做端到端评估，建立可重复跑的质量基线。

## Key Changes
- 查询处理：
  - 新增 `QueryNormalizer`：输出 `original_query`、`normalized_query`、`medical_terms`、`rewrite_reason`。
  - KG 和 Milvus 检索使用 `normalized_query`，回答仍基于用户原始问题表达。
  - `QueryRouter` 增加 `needs_case_context`、`answer_style`，用于决定是否查个人病例库和选择回答模板。

- 检索路由：
  - 保留 Neo4j KG 的现有 15 类/16 类意图识别，用于结构化查询。
  - Toyhom/Milvus 改用规范化 query 提升召回。
  - 新增个人病例检索源，仅按当前用户过滤。
  - 修复上传入库字段问题：`insert_batch()` 支持 `id` 或 `pk`，避免病例向量化被静默跳过。

- 回答 prompt：
  - 替换强制五层结构。
  - 按 `answer_style` 选择模板：事实短答、用药边界、症状鉴别、检查报告、饮食/科室、个人病例解读。
  - 个人病例回答必须区分“病例中已有信息”和“通用医学建议”。
  - 安全免责声明继续由 `SafetyGuard` 统一追加，避免 prompt 和 guard 重复提醒。

- 用户病例：
  - 上传后保存原始文件、脱敏摘要、病例切块向量。
  - 文档索引增加 `username`、`document_id`、`summary`、`chunk_count`、`uploaded_at`、`status`。
  - 病例向量数据必须带 `source_type="user_case"`、`username`、`document_id`、`filename`。
  - 若当前 Milvus collection 不适合新增字段，新增独立 `user_cases` collection。

## RAGAS Evaluation
- 新增 `eval/` 评估模块：
  - `eval/golden_cases.yaml` 或 `.jsonl`：维护 30-50 条人工小金集。
  - 覆盖 KG 查询、Toyhom 向量查询、混合检索、低召回场景、个人病例问答、高风险安全场景。
  - 每条样本包含 `question`、`ground_truth`、`expected_context_keywords`、可选 `case_context` 和 `expected_route`。

- 端到端评估流程：
  - 调用真实 RAG pipeline 获取 `answer`、`retrieved_contexts`、`route`。
  - 用 RAGAS 计算 `faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`。
  - 额外记录自定义指标：路由准确率、病例隔离命中率、安全提醒命中率、空检索率。
  - 输出 `eval/reports/ragas_report.json` 和简洁表格摘要。

- 质量门槛：
  - 第一版先建立 baseline，不强制阻塞开发。
  - 后续可设置回归阈值，例如 faithfulness 或 answer_relevancy 明显下降时 CI 失败。

## Test Plan
- 单元测试：
  - `QueryNormalizer` 对口语问题输出稳定医学规范化 query。
  - `QueryRouter` 正确输出 `query_type`、`answer_style`、病例检索开关。
  - `PromptBuilder` 验证不同意图模板，不再验证固定五段式。
  - `MilvusClientWrapper.insert_batch()` 同时支持 `id` 和 `pk`。
  - 文档索引和病例检索按用户隔离。

- 集成测试：
  - 上传病例后生成摘要、写索引、写向量 chunk。
  - 聊天时当前用户病例摘要和病例检索结果进入 prompt。
  - 普通医学问题不强行引用病例。
  - 删除病例同步删除文件、索引、向量 chunk。
  - RAGAS 评估脚本能在小金集上跑通并生成报告。

## Assumptions
- 病例存储采用“个人病例库”，不污染公共 Toyhom 问答库。
- 查询翻译优先做“中文医学规范化”，暂不做完整中英翻译。
- 回答风格采用“按意图模板”。
- RAGAS 第一版使用 30-50 条人工小金集，评估范围为端到端 RAG。
