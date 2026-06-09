# MedAgent 升级计划：从 RAG 流水线到状态化 AI Agent

> **目标：** 将当前的医疗 RAG 问答系统升级为状态化 AI Agent（MedAgent），引入记忆系统、智能路由、ReAct 推理和架构级弹性，最终形态接近 AGI-saber 模式。

**核心理念：** 系统从"查询-检索-生成"的流水线，进化为"感知-记忆-推理-行动"的智能体循环。

---

## 当前架构分析（as-is）

### 一、整体架构分层

```
┌─────────────────────────────────────────────────────────────────┐
│              Frontend (静态文件, HTML/JS)                         │
├─────────────────────────────────────────────────────────────────┤
│                    FastAPI 路由层                                 │
│  /auth  /chat/stream  /sessions  /documents  /health            │
├─────────────────────────────────────────────────────────────────┤
│             MedicalChatService (编排中枢)                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Tool 匹配 │  │ QueryRouter  │  │ 记忆系统  │  │ SafetyGuard│  │
│  │ (快速路径)│  │ (4 种模式)   │  │          │  │ (风险分级) │  │
│  └──────────┘  └──────┬───────┘  └──────────┘  └────────────┘  │
│                        ▼                                         │
│            执行模式: Tool / Chat / RAG / ReAct                    │
├─────────────────────────────────────────────────────────────────┤
│                   检索层 (HybridRetriever)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ KGRetriever│  │ QARetriever│  │ ESBM25    │  │QueryNormalizer│
│  │ (Neo4j KG)│  │(Milvus ANN)│  │(BM25 关键词)│+ Intent      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
│                       │ RRF 融合 + Cross-Encoder 重排序           │
│                       ▼                                          │
│              PromptBuilder + ContextAssembler                    │
│         (Schema-Driven Context Assembly, 优先级 + Token 预算)    │
├─────────────────────────────────────────────────────────────────┤
│                    生成层 (LLM Provider)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ DeepSeek  │  │ ZhipuAI  │  │ Qwen     │  │ Ollama   │       │
│  │(API/官方) │  │ (智谱)   │  │ (通义)   │  │ (本地)   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                    基础设施                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ 健康追踪  │  │ JSON持久化 │  │ NER 模型 │  │ 认证系统 │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 二、核心模块详解

#### 1. 路由层 — `src/medrag/app/`

| 文件 | 职责 |
|------|------|
| `server.py` | FastAPI 应用入口。`lifespan` 异步初始化 `MedicalChatService`，挂载 4 组路由 + 静态文件 + `/health` 端点 |
| `api/chat.py` | SSE 流式聊天 `POST /chat/stream`，`ThreadPoolExecutor` 包装阻塞生成器为异步事件流，支持运行时切换 LLM 提供商与模型 |
| `api/auth.py` | 用户注册/登录/鉴权 |
| `api/sessions.py` | 会话 CRUD（JSON 文件持久化） |
| `api/documents.py` | 用户上传文档管理 |
| `session_store.py` | 会话消息存储，每条对话存为 `{role, content}` 列表 |
| `schemas.py` | Pydantic 模型（`ChatRequest`, `ModelsResponse` 等） |

#### 2. 服务编排层 — `src/medrag/service/chat_service.py`

**`MedicalChatService`** 是整个系统的中枢，`chat()` / `stream_chat()` 统一入口，按优先级分发四种执行模式：

| 模式 | 触发条件 | 流程 |
|------|---------|------|
| **Tool** | `ToolRegistry.match()` 命中内置工具 | 直接执行工具，返回结构化结果 |
| **Chat** | 问候/闲聊（Router 判定 `execution_mode=chat`） | 直接 LLM 回复，不走检索 |
| **RAG** | 通用医疗问题（默认 `execution_mode=rag`） | 检索 → 重排序 → Context Assembly → 生成 → 安全检查 |
| **ReAct** | 复杂多步问题（`execution_mode=react`） | Thought/Action/Observation 循环 |

**Tool 优先策略**：`chat()` 第一步始终检查 `ToolRegistry.match()`，命中则立即返回，完全不经过 Router / Retrieval / LLM。

#### 3. 检索层 — `src/medrag/retrieval/` + `src/medrag/vectors/`

**`HybridRetriever`** 编排三个检索后端：

| 检索器 | 后端 | 算法 | 数据源 |
|--------|------|------|--------|
| `KGRetriever` | Neo4j | NER → 意图识别 → Cypher 查询 | DiseaseKG（8 类实体、11 类关系、44K 实体、312K 关系） |
| `QARetriever` | Milvus / Zilliz Cloud | BGE Embedding + COSINE ANN | cMedQA2 数据集 |
| `ESBM25Retriever` | Elasticsearch | BM25（`question^3` + `title^2` + `answer` 加权） | cMedQA2 数据集 |

**融合策略**：Milvus Dense + ES Sparse → **RRF (Reciprocal Rank Fusion, c=60)** → **Cross-Encoder 重排序** → top-k

**`QueryRouter`**（双模路由）：
- **LLM 路由优先**：DeepSeek 语义分类，输出 `execution_mode`、`use_kg`、`use_qa`、`query_type`、`needs_case_context`
- **规则路由回退**：关键词匹配（问候/ReAct/药品/饮食/科室等 8 类），零成本无依赖

**`QueryNormalizer`**：查询规范化（同义词映射、医学术语对齐）

#### 4. 记忆系统 — `src/medrag/memory/`

借鉴 AGI-saber 记忆架构，4 层集成：

| 层级 | 实现 | 功能 |
|------|------|------|
| **STM**（短期） | `ShortTermMemory` | 滑动窗口，保留最近 5 轮对话 |
| **LTM**（长期） | `LongTermMemory` | Embedding 余弦召回（`cosine × 0.7 + importance × 0.3`），TF-IDF 回退；支持 JSON 自动持久化 |
| **Preference** | `PreferenceStore` | 规则提取用户偏好（姓名、年龄等） |
| **GraphMemory** | `GraphMemory` | LTM 的图感知封装，支持关联扩展和 consolidation |

**Consolidation**（记忆巩固）三阶段：衰减 → 去重+合并 → 过期清理，循环触发。

**Schema-Driven Context Assembly**（`memory/schema.py`）：
- `ContextAssembler` 按优先级插槽管理 Token 预算
- 优先级：Memory(100) > CaseSummary(90) > KG(80) > QA(70) > CaseChunks(60) > Query(50)
- 超预算时从最低优先级开始丢弃

#### 5. ReAct 多步推理引擎 — `src/medrag/react/`

| 组件 | 职责 |
|------|------|
| `engine.py` | Thought/Action/Observation 循环，最大 6 步，格式错误自动引导重试 |
| `tools.py` | `ReActTool` 定义 |

注册的工具：`retrieve_kg`（KG 搜索）、`retrieve_qa`（QA + KG 联合搜索）、`calculate_dosage`（剂量计算）、`query_normal_range`（正常值查询）

#### 6. 工具系统 — `src/medrag/tools/`

所有工具实现 `BaseTool` 接口（`match()` + `execute()`），`ToolRegistry` 管理注册和匹配：

| 工具 | match 策略 | execute 逻辑 |
|------|-----------|-------------|
| `DosageCalculator` | 正则提取药物 + 年龄/体重 | 剂量公式计算（mg/kg） |
| `DepartmentGuide` | 科室关键词匹配 | 返回科室建议文本 |
| `NormalRangeTool` | 检查项关键词匹配 | 返回正常值参考范围，含偏高/偏低判断 |

#### 7. RAG 流水线 — `src/medrag/rag/`

| 模块 | 核心逻辑 |
|------|---------|
| `PromptBuilder` | **双层设计**：Tier 1（检索充足 → 基于资料 + 标注来源），Tier 2（无检索 → 固定降级话术"建议咨询医生"） |
| `AnswerGenerator` | 同步生成 + 流式生成，支持动态切换 provider/model |
| `SafetyGuard` | **双色风险分级**：红色（立即急诊，开头插入警告）、黄色（尽快就医，末尾插入提醒）；按 query_type 决定是否附免责声明 |

#### 8. LLM 集成 — `src/medrag/llm/`

工厂模式创建并缓存客户端（`get_llm_client()`）：

| 提供商 | SDK | 默认模型 |
|--------|-----|---------|
| DeepSeek | `openai.OpenAI` | deepseek-chat |
| ZhipuAI | `zhipuai.ZhipuAI` | glm-4-plus |
| Qwen | `openai.OpenAI`（DashScope 兼容端点） | qwen-plus |
| Ollama | `openai.OpenAI`（本地端点） | qwen3.5:9b |

#### 9. 基础设施

| 模块 | 功能 |
|------|------|
| `infrastructure/health.py` | 全局组件健康注册表，追踪 ES/Neo4j/Milvus 状态 |
| `infrastructure/ner/` | NER 模型加载器 |
| `infrastructure/storage/json_store.py` | JSON 文件持久化基类 |
| `data/text_cleaner.py` | 文本清洗 |
| `data/case_parser.py` | 用户病例解析 |
| `ner/model.py` | RoBERTa + BiLSTM NER 模型 |
| `config/settings.py` | 从环境变量 + `.env` 读取的不可变配置类 |

### 三、一次典型 RAG 请求的数据流

```
用户输入 → auth 鉴权 → POST /chat/stream (SSE)
                          │
  ┌───────────────────────▼───────────────────────┐
  │      MedicalChatService.stream_chat()          │
  │                                               │
  │  (1) ToolRegistry.match(query)                │
  │      └─ 命中工具? → 直接执行并返回             │
  │                                               │
  │  (2) QueryRouter.route(query)                 │
  │      └─ LLM 语义 / 规则回退                   │
  │         ├─ execution_mode: chat/tool/rag/react│
  │         ├─ use_kg: bool                       │
  │         ├─ use_qa: bool                       │
  │         └─ query_type: 8 类                   │
  │                                               │
  │  (3) SafetyGuard.detect_risk(query)           │
  │      └─ red / yellow / none                   │
  │                                               │
  │  (4) HybridRetriever.retrieve(query)          │
  │      ├─ QueryNormalizer.normalize()           │
  │      ├─ KGRetriever.search() → Neo4j          │
  │      ├─ QARetriever.search() → Milvus ANN     │
  │      ├─ ESBM25Retriever.search() → ES BM25    │
  │      ├─ RRF 融合 (Milvus + ES)               │
  │      └─ UserCaseRetriever.search() (可选)     │
  │                                               │
  │  (5) Reranker.rerank() → Cross-Encoder 精排   │
  │                                               │
  │  (6) MemorySystem.build_context()             │
  │      └─ STM 最近轮次 + LTM 语义召回 + 偏好    │
  │                                               │
  │  (7) ContextAssembler.assemble()              │
  │      └─ 优先级排序 → Token 预算裁剪          │
  │                                               │
  │  (8) PromptBuilder.build_messages_with_context│
  │      └─ System prompt + 检索上下文 + 用户问题 │
  │                                               │
  │  (9) AnswerGenerator.generate_stream()        │
  │      └─ DeepSeek / ZhipuAI / Qwen / Ollama    │
  │                                               │
  │ (10) SafetyGuard.append_safety_notice()       │
  │      └─ 急诊警告 / 就医提醒 / 免责声明        │
  │                                               │
  │ (11) MemorySystem.store_assistant_reply()     │
  │      └─ 去重 → 持久化 (auto-save)             │
  └───────────────────────────────────────────────┘
```

### 四、架构特性总结

| 特性 | 实现方式 |
|------|---------|
| **多模式路由** | LLM 语义路由（主） + 规则路由（备），4 种执行模式 |
| **多引擎检索** | KG（结构化）+ Milvus ANN（语义）+ ES BM25（稀疏）三路并行 |
| **RRF 融合** | Dense + Sparse 倒数排名融合，跨源分数叠加 |
| **Cross-Encoder 精排** | 对 RRF 结果二次排序，提升 top-k 精准度 |
| **Schema-Driven 上下文** | 优先级插槽 + 全局 Token 预算裁剪 |
| **分层记忆** | STM + LTM + Preference + GraphMemory，带 consolidation |
| **ReAct 推理** | Thought/Action/Observation 循环，6 步上限，工具注册制 |
| **工具优先** | Tool 快速路径绕过整个 RAG 流水线 |
| **优雅降级** | 每个外部依赖独立 try/except，health.py 全局追踪 |
| **分级安全** | 红色急诊 + 黄色就医 + 检索质量免责声明 |
| **多 LLM 提供商** | 4 个后端，工厂模式缓存，运行时切换 |
| **流式 SSE** | ThreadPoolExecutor + asyncio.Queue 包装为异步事件流 |

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