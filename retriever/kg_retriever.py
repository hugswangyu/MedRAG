"""Neo4j Knowledge Graph Retriever.

Wraps the project's existing NER model, DeepSeek intent recognition, and
Cypher query patterns into a single KGRetriever class with a unified
``search(query) -> List[Dict]`` interface.

Does NOT rewrite the original logic — NER is imported from ``ner_model``,
intent recognition reuses the same prompt as ``webui.Intent_Recognition``,
and Cypher patterns mirror ``webui.generate_prompt``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import py2neo
from openai import OpenAI

from config.settings import settings
import ner_model as zwk

# ---------------------------------------------------------------------------
# Intent → Cypher mapping
# ---------------------------------------------------------------------------
# Each tuple: (keyword, query_type, relation_or_attribute, target_type, required_entity_type)
# query_type: "attribute" | "relation" | "reverse_relation"
#
# NOTE: keyword order matters. "治疗周期" must be checked before "治疗" to
# avoid false matches; "查询疾病所属科目" is a full sentence-level check
# (see webui.py:217).  This list mirrors the if/elif chain in
# webui.generate_prompt exactly.
# ---------------------------------------------------------------------------
_INTENT_SPEC: List[tuple] = [
    ("简介",           "attribute",        "疾病简介",       None,       "疾病"),
    ("病因",           "attribute",        "疾病病因",       None,       "疾病"),
    ("预防",           "attribute",        "预防措施",       None,       "疾病"),
    ("治疗周期",       "attribute",        "治疗周期",       None,       "疾病"),
    ("治愈概率",       "attribute",        "治愈概率",       None,       "疾病"),
    ("易感人群",       "attribute",        "疾病易感人群",   None,       "疾病"),
    ("药品",           "relation",         "疾病使用药品",   "药品",     "疾病"),
    ("宜吃食物",       "relation",         "疾病宜吃食物",   "食物",     "疾病"),
    ("忌吃食物",       "relation",         "疾病忌吃食物",   "食物",     "疾病"),
    ("检查项目",       "relation",         "疾病所需检查",   "检查项目", "疾病"),
    ("查询疾病所属科目", "relation",        "疾病所属科目",   "科目",     "疾病"),
    ("症状",           "relation",         "疾病的症状",     "疾病症状", "疾病"),
    ("治疗",           "relation",         "治疗的方法",     "治疗方法", "疾病"),
    ("并发",           "relation",         "疾病并发疾病",   "疾病",     "疾病"),
    ("生产商",         "reverse_relation", "生产",           "药品商",   "药品"),
]


class KGRetriever:
    """Unified retriever over the Neo4j medical knowledge graph.

    Relies on the **existing** NER pipeline (``ner_model``) and replicates
    the Cypher patterns from ``webui.generate_prompt``.  Intent recognition
    uses DeepSeek with the same few-shot prompt as ``webui.Intent_Recognition``.

    Usage::

        # Obtain NER components (same as webui.load_model)
        rule = zwk.rule_find()
        tfidf_r = zwk.tfidf_alignment()
        ...

        retriever = KGRetriever(
            bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag,
        )
        results = retriever.search("感冒了怎么办")
        # results is List[Dict], each dict has:
        #   source, intent, entity, relation, answer, evidence, score
    """

    def __init__(
        self,
        bert_model,
        bert_tokenizer,
        rule,
        tfidf_r,
        device,
        idx2tag,
        neo4j_client: Optional[py2neo.Graph] = None,
    ):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.rule = rule
        self.tfidf_r = tfidf_r
        self.device = device
        self.idx2tag = idx2tag

        self.neo4j = neo4j_client or self._create_neo4j_client()

        # DeepSeek client for intent recognition.
        # TODO: If you refactor Intent_Recognition out of webui.py into a
        # shared module (e.g. intent.py), replace this inline client with an
        # import of that shared function.
        self.llm = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_neo4j_client() -> py2neo.Graph:
        return py2neo.Graph(
            settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            name=settings.neo4j_database,
        )

    def _get_entities(self, query: str) -> Dict[str, str]:
        """NER pipeline: {entity_type: canonical_name}."""
        try:
            return zwk.get_ner_result(
                self.bert_model,
                self.bert_tokenizer,
                query,
                self.rule,
                self.tfidf_r,
                self.device,
                self.idx2tag,
            )
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Intent recognition
    # ------------------------------------------------------------------
    # The prompt below is copied from webui.Intent_Recognition to avoid
    # importing Streamlit-dependent modules.  If you extract intent
    # recognition into a shared utility, import it here instead.
    # ------------------------------------------------------------------

    _INTENT_PROMPT_TEMPLATE = """阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

**查询类别**
- "查询疾病简介"
- "查询疾病病因"
- "查询疾病预防措施"
- "查询疾病治疗周期"
- "查询治愈概率"
- "查询疾病易感人群"
- "查询疾病所需药品"
- "查询疾病宜吃食物"
- "查询疾病忌吃食物"
- "查询疾病所需检查项目"
- "查询疾病所属科目"
- "查询疾病的症状"
- "查询疾病的治疗方法"
- "查询疾病的并发疾病"
- "查询药品的生产商"

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题。
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
- 确保最终的输出列表包含了所有与用户问题相关的类别描述。

以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
**示例1：**
输入："睡眠不好，这是为什么？"
输出：["查询疾病简介","查询疾病病因"]  # 这个问题隐含地询问了睡眠不好的病因
**示例2：**
输入："感冒了，怎么办才好？"
输出：["查询疾病简介","查询疾病所需药品", "查询疾病的治疗方法"]  # 用户可能既想知道应该吃哪些药品，也想了解治疗方法
**示例3：**
输入："跑步后膝盖痛，需要吃点什么？"
输出：["查询疾病简介","查询疾病宜吃食物", "查询疾病所需药品"]  # 这个问题可能既询问宜吃的食物，也可能在询问所需药品
**示例4：**
输入："我怎样才能避免冬天的流感和感冒？"
输出：["查询疾病简介","查询疾病预防措施"]  # 询问的是预防措施，但因为提到了两种疾病，这里隐含的是对共同预防措施的询问
**示例5：**
输入："头疼是什么原因，应该怎么办？"
输出：["查询疾病简介","查询疾病病因", "查询疾病的治疗方法"]  # 用户询问的是头疼的病因和治疗方法
**示例6：**
输入："如何知道自己是不是有艾滋病？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有艾滋病，一定一定要进行相关检查，这是根本性的！其次是查看疾病的病因，看看自己的行为是不是和病因重合。
**示例7：**
输入："我该怎么知道我自己是否得了21三体综合症呢？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有21三体综合症，一定一定要进行相关检查(比如染色体)，这是根本性的！其次是查看疾病的病因。
**示例8：**
输入："感冒了，怎么办？"
输出：["查询疾病的治疗方法","查询疾病所需药品","查询疾病所需检查项目","查询疾病宜吃食物"]  # 问怎么办，首选治疗方法。然后是要给用户推荐一些药，最后让他检查一下身体。同时，也推荐一下食物。
**示例9：**
输入："癌症会引发其他疾病吗？"
输出：["查询疾病的并发疾病"]  # 显然，用户问的是疾病并发疾病，随后可以给用户科普一下癌症简介。
**示例10：**
输入："葡萄糖浆的生产者是谁？葡萄糖浆是谁生产的？"
输出：["查询药品的生产商"]  # 显然，用户想要问药品的生产商
通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。

**注意：**
- 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述5个示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。

现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
问题输入："{query}"
输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！确保你的解释和合乎逻辑的！注意，如果用户询问了有关疾病的问题，一般都要先介绍一下疾病，也就是有"查询疾病简介"这个需求。
再次检查你的输出都包含在**查询类别**:"查询疾病简介"、"查询疾病病因"、"查询疾病预防措施"、"查询疾病治疗周期"、"查询治愈概率"、"查询疾病易感人群"、"查询疾病所需药品"、"查询疾病宜吃食物"、"查询疾病忌吃食物"、"查询疾病所需检查项目"、"查询疾病所属科目"、"查询疾病的症状"、"查询疾病的治疗方法"、"查询疾病的并发疾病"、"查询药品的生产商"。
"""  # noqa: E501 — few-shot prompt is intentionally long

    def _recognize_intents(self, query: str) -> str:
        """Call DeepSeek for intent recognition.

        Returns the raw API response string (e.g.
        ``["查询疾病简介","查询疾病病因"] # comment``), or ``""`` on failure.
        """
        try:
            prompt = self._INTENT_PROMPT_TEMPLATE.format(query=query)
            response = self.llm.chat.completions.create(
                model=settings.deepseek_default_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Cypher query helpers
    # ------------------------------------------------------------------
    # These replicate the add_shuxing_prompt / add_lianxi_prompt functions
    # from webui.py but return raw data instead of prompt strings.
    # ------------------------------------------------------------------

    def _query_attribute(self, entity: str, attribute: str) -> Optional[str]:
        """``MATCH (a:疾病{{名称:'...'}}) RETURN a.<attribute>``"""
        cypher = "match (a:疾病{名称:'%s'}) return a.%s" % (entity, attribute)
        try:
            row = self.neo4j.run(cypher).data()[0]
            values = list(row.values())
            if values:
                return "".join(str(v) for v in values if v)
        except Exception:
            pass
        return None

    def _query_relation(
        self, entity: str, relation: str, target_type: str
    ) -> Optional[List[str]]:
        """``MATCH (a:疾病{{名称:'...'}})-[r:REL]->(b:TYPE) RETURN b.名称``"""
        cypher = (
            "match (a:疾病{名称:'%s'})-[r:%s]->(b:%s) return b.名称"
            % (entity, relation, target_type)
        )
        try:
            rows = self.neo4j.run(cypher).data()
            return [list(r.values())[0] for r in rows if r.values()]
        except Exception:
            pass
        return None

    def _query_reverse_relation(
        self, entity: str, relation: str, source_type: str
    ) -> Optional[List[str]]:
        """``MATCH (a:SOURCE)-[r:REL]->(b:药品{{名称:'...'}}) RETURN a.名称``"""
        cypher = (
            "match (a:%s)-[r:%s]->(b:药品{名称:'%s'}) return a.名称"
            % (source_type, relation, entity)
        )
        try:
            rows = self.neo4j.run(cypher).data()
            return [list(r.values())[0] for r in rows if r.values()]
        except Exception:
            pass
        return None

    def _resolve_disease_from_symptom(self, symptom: str) -> Optional[str]:
        """Reverse-lookup: symptom → possible diseases, pick one at random."""
        cypher = (
            "match (a:疾病)-[r:疾病的症状]->(b:疾病症状 {名称:'%s'}) return a.名称"
            % symptom
        )
        try:
            rows = self.neo4j.run(cypher).data()
            names = [list(r.values())[0] for r in rows if r.values()]
            if names:
                return random.choice(names)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        intents: Optional[str] = None,
    ) -> List[Dict]:
        """Search the Neo4j knowledge graph for information relevant to *query*.

        Parameters
        ----------
        query:
            Natural-language medical question.
        intents:
            Raw intent recognition result.  When *None* (default), intents
            are auto-detected via DeepSeek.  You can pass a pre-computed
            string (the output of ``Intent_Recognition``) to skip the LLM
            call.

        Returns
        -------
        List[Dict]
            Each dict has keys ``source``, ``intent``, ``entity``,
            ``relation``, ``answer``, ``evidence``, ``score``.
            Returns an empty list when no entity is found or all queries
            come back empty.
        """
        # 1. NER
        entities = self._get_entities(query)
        if not entities:
            return []

        # 2. Intent recognition (cached if caller provided it)
        raw_intents = intents if intents is not None else self._recognize_intents(query)
        if not raw_intents:
            return []

        # 3. Special case: symptom without disease → reverse-lookup
        disease = entities.get("疾病")
        if "疾病症状" in entities and disease is None:
            disease = self._resolve_disease_from_symptom(entities["疾病症状"])

        # 4. Execute matched intents
        results: List[Dict] = []
        for keyword, qtype, rel_attr, target, req_entity in _INTENT_SPEC:
            if keyword not in raw_intents:
                continue

            entity = entities.get(req_entity)
            if entity is None:
                continue

            evidence = None
            answer_parts: List[str] = []

            if qtype == "attribute":
                value = self._query_attribute(entity, rel_attr)
                if value:
                    evidence = value
                    answer_parts.append(value)

            elif qtype == "relation":
                names = self._query_relation(entity, rel_attr, target)  # type: ignore[arg-type]
                if names:
                    evidence = names
                    answer_parts.append("、".join(names))

            elif qtype == "reverse_relation":
                names = self._query_reverse_relation(entity, rel_attr, target)  # type: ignore[arg-type]
                if names:
                    evidence = names
                    answer_parts.append("、".join(names))

            if evidence is not None:
                results.append(
                    {
                        "source": "neo4j_kg",
                        "intent": keyword,
                        "entity": entity,
                        "relation": rel_attr,
                        "answer": "".join(answer_parts),
                        "evidence": evidence,
                        "score": 1.0,
                    }
                )

        return results
