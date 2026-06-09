# MedAgent 升级计划：从 RAG 流水线到状态化 AI Agent

> **目标：** 将当前的医疗 RAG 问答系统升级为状态化 AI Agent（MedAgent），引入记忆系统、智能路由、ReAct 推理和架构级弹性，最终形态接近 AGI-saber 模式。

**核心理念：** 系统从"查询-检索-生成"的流水线，进化为"感知-记忆-推理-行动"的智能体循环。

---

## 整体架构迁移

```
当前:   Query → Router → Retrieve → Rerank → Prompt → Generate → Safety
                                                                    
迁移后:  Query → Memory Recall → Context Assembly → ReAct Loop → Action → Memory Store
                                          ↑                           ↑
                                     Schema-Driven              Tool Execution
```

---

## Phase 1: 记忆系统（Memory System）

**状态：** ✅ 已有完整实现方案，8 个任务

**目标：** 为系统添加三层记忆（STM/LTM/Graph + Preference），使每次对话能记住用户信息和历史。

**组件：**
| 层 | 存储 | 功能 |
|----|------|------|
| ShortTermMemory | 内存滑动窗口 | 最近 N 轮对话 |
| LongTermMemory | NumPy + Embedding/TF | 跨会话语义召回 |
| GraphMemory | Neo4j（可选） | FOLLOWS/SIMILAR_TO 关联扩展 |
| PreferenceStore | 内存 KV | 用户偏好（姓名、城市等） |

**集成方式：** `MemorySystem` 门面类注入 `MedicalChatService`，在 chat() 入口调用 `memory.recall()` 注入上下文，退出时调用 `memory.add_message()` 和 `remember()`。

详见：`docs/superpowers/plans/2026-06-08-memory-system.md`

---

## Phase 2: 智能路由与 Schema-Driven 上下文装配

**状态：** 📝 待规划

### 2.1 路由升级（多模式路由）

当前 `QueryRouter` 只做源选择（KG/QA/无），升级后支持 4 种执行模式：

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| **Chat** | 闲聊/问候 | 直接 LLM 回复，不走 RAG |
| **Tool** | 简单指令（查天气、算剂量） | 单步工具调用，无 RAG |
| **RAG** | 医学知识查询 | 完整 RAG 流水线（当前行为） |
| **ReAct** | 复杂/多步推理需求 | 循环推理 -> 工具 -> 观察 -> 推理 |

实现思路：给 `QueryRouter` 增加 `execution_mode` 输出，`chat_service` 根据模式分发到不同执行路径。

### 2.2 Schema-Driven Context Assembly

参考 AGI-saber 的 `RuntimeContextSchema` + `ContextAssembler` 模式：

```python
@dataclass
class Slot:
    kind: str          # kg / qa / case / memory 等
    content: str | None
    priority: int      # 裁剪优先级
    token_count: int

class ContextAssembler:
    """并发填充 Slot，全局 Token 预算裁剪。"""
    slots: List[Slot]
    budget: int = 4096  # context window
    
    async def assemble(self) -> List[Slot]:
        # 1. 并发填充所有 Slot
        # 2. 按优先级裁剪到 budget
```

**当前状态分析：** 代码中已经有 `build_messages()` 和 `_build_tier1_parts()` 做类似的事情（分 section 组装上下文），但缺少：
- 显式的 slot 定义和优先级声明
- 全局 token 预算计算和裁剪
- 并发填充（当前是顺序）

**迁移策略：** 在现有 `PromptBuilder` 基础上逐步引入 Slot 抽象，先改为显式 Slot 列表，再加入 budget 检查，最后改为并发填充。

---

## Phase 3: ReAct 多步推理

**状态：** 📝 待规划

### 3.1 ReAct 循环

引入 Planner LLM + Generator LLM 双层架构：

```
用户问题
   ↓
Router 判定为 ReAct 模式
   ↓
┌─ ReAct Loop ──────────────────────────┐
│  Thought（Planner LLM）                │
│  Action（工具调用/检索）                │
│  Observation（结果注入上下文）           │
│  → 继续或结束                           │
└────────────────────────────────────────┘
   ↓
Generator LLM 生成最终回答
   ↓
记忆提取与持久化
```

### 3.2 工具注册

| 工具名 | 对应当前组件 | 说明 |
|--------|-------------|------|
| `retrieve_kg` | KGRetriever | 知识图谱查询 |
| `retrieve_qa` | QARetriever + ESBM25Retriever | QA 向量库 + BM25 |
| `rerank` | CrossEncoderReranker | 重排序 |
| `calculate_dosage` | 新开发 | 剂量计算（可选） |
| `web_search` | 新开发 | 联网搜索（可选） |

### 3.3 与现有系统的关系

ReAct 模式不替换现有 RAG 流水线，而是作为 RAG 的上层控制器：
- 简单查询 → 直接走 RAG（现有逻辑）
- 复杂查询（需多步推理、多工具组合）→ ReAct Loop
- 两种路径共用同一套检索工具

---

## Phase 4: 架构弹性与优雅降级

**状态：** 📝 待规划

**目标：** 系统不依赖任何单一基础设施组件，每个组件故障时自动降级。

| 组件 | 降级策略 |
|------|----------|
| Milvus | → ES BM25 单源 → SimpleReranker → 仅 KG |
| ES | → Milvus 单源 |
| KG（Neo4j） | → 跳过 KG，仅用 QA |
| LLM API | → 缓存回答 → Ollama 本地模型 |
| CrossEncoder | → SimpleReranker（关键词命中） |
| Embedding Model | → TF-IDF 回退 |

**当前状态：** 部分已有（`get_reranker()` 有 CE → Simple 回退链，各 retriever 有 `_safe_*_search` try/except），需要系统化。

---

## 执行路线图

| Phase | 内容 | 预估工作量 | 前置条件 |
|-------|------|-----------|---------|
| **1** | 记忆系统（8 任务） | 3-5 天 | 无 |
| **2a** | 路由升级（多模式） | 2-3 天 | Phase 1 |
| **2b** | Schema-Driven Context Assembly | 2-3 天 | Phase 1 |
| **3** | ReAct 多步推理 | 5-7 天 | Phase 2 |
| **4** | 架构弹性 | 2-3 天 | 可并行 |

**总预估：** 14-21 天

---

## 当前已完成的工作（本次 Session）

- [x] cMedQA2 数据集替换（Loader + 双索引重建中）
- [x] RRF 交叉叠加 bug 修复（c: 60→20，跨源累加）
- [x] Pipeline 参数调优（retrieval_top_k=15, rerank_top_k=8）
- [x] CrossEncoder 保留 RRF 分数
- [x] 删除所有旧 Toyhom 代码和索引
- [x] `ToyhomQARetriever` → `QARetriever`
- [x] 全局变量重命名（toyhom → qa）
