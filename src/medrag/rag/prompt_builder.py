"""提示词构建器：将检索结果组装为最终的 LLM 提示词。"""

from __future__ import annotations

from typing import Dict, List, Optional

from medrag.prompts import (
    MEDICAL_ANSWER_PROMPT,
    QUERY_TYPE_HINTS,
    ANSWER_STYLE_HINTS,
    CONTEXT_CASE_HEADER,
    CONTEXT_USER_CASE_CHUNKS_HEADER,
    CONTEXT_KG_HEADER,
    CONTEXT_TOYHOM_HEADER,
    CONTEXT_EMPTY_NOTE,
)

MAX_PER_SOURCE = 5          # 每个来源在提示词中的最大结果数
MAX_RESULT_CHARS = 400      # 每条结果截断长度

# 检索置信度指令
_CONFIDENCE_NONE = (
    "\n\n⚠️ **系统提示**：知识库中未检索到直接相关资料。"
    '请基于通用医学知识回答，并在⑤中标注“该信息未在知识库中检索到，'
    '基于通用医学知识提供参考，请务必核实”。'
    '绝对禁止因缺少检索资料而拒答基础医学事实。'
)
_CONFIDENCE_LOW = (
    '\n\n⚠️ **系统提示**：检索到的资料置信度较低或存在矛盾。'
    '回答时请在⑤中标注“不同资料对此存在差异，请以医生意见为准”。'
)
_CONFIDENCE_NONE_NOTE = '\n\n⚠️ 知识库中未检索到直接相关资料，请基于通用医学知识回答，并标注不确定性。'


class PromptBuilder:
    """将多源检索结果组装为完整的答案生成提示词。

    用法::

        builder = PromptBuilder()
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            kg_results=kg_results,
            toyhom_results=toyhom_results,
            case_context=None,
            route=route,
            retrieval_quality={"has_kg": True, "has_qa": False, "confidence": "high"},
        )
        # → 将 *prompt* 喂给 DeepSeek / OpenAI
    """

    def build_answer_prompt(
        self,
        query: str,
        kg_results: Optional[List[Dict]] = None,
        toyhom_results: Optional[List[Dict]] = None,
        case_results: Optional[List[Dict]] = None,
        case_context: Optional[str] = None,
        route: Optional[Dict] = None,
        retrieval_quality: Optional[Dict] = None,
        query_info: Optional[Dict] = None,
    ) -> str:
        """构建用于回答 LLM 的最终提示词字符串。

        Args:
            query: 用户原始问题。
            kg_results: KGRetriever.search() 输出（已重排序）。
            toyhom_results: ToyhomQARetriever.search() 输出（已重排序）。
            case_context: 预先计算的用户病例摘要，或 None。
            route: 路由器决策字典，query_type 用于注入分类指令。
            retrieval_quality: {"has_kg": bool, "has_qa": bool, "confidence": "high"/"low"/"none"}。
        """
        # --- 根据 query_type 注入分类指令 ---
        query_type = (
            route.get("query_type", "general_medical_qa")
            if route else "general_medical_qa"
        )
        type_hint = QUERY_TYPE_HINTS.get(query_type, QUERY_TYPE_HINTS["general_medical_qa"])
        answer_style = (route or {}).get("answer_style", "general_guidance")
        if case_context or case_results:
            answer_style = "case_based"
        style_hint = ANSWER_STYLE_HINTS.get(answer_style, ANSWER_STYLE_HINTS["general_guidance"])
        system_prompt = (
            MEDICAL_ANSWER_PROMPT
            + "\n\n## 当前问题类型\n"
            + type_hint
            + "\n\n## 当前回答风格\n"
            + style_hint
        )

        # --- 检索置信度指令 ---
        confidence_note = ""
        if retrieval_quality:
            has_any = retrieval_quality.get("has_kg") or retrieval_quality.get("has_qa")
            conf = retrieval_quality.get("confidence", "high")
            if not has_any or conf == "none":
                confidence_note = _CONFIDENCE_NONE_NOTE
            elif conf == "low":
                confidence_note = _CONFIDENCE_NONE_NOTE

        # --- 组装上下文块 ---
        sections: list[str] = []

        # 1. 病例上下文（最高优先级）
        if case_context:
            sections.append(
                CONTEXT_CASE_HEADER.format(case_text=case_context.strip())
            )
        else:
            sections.append(
                CONTEXT_CASE_HEADER.format(case_text=CONTEXT_EMPTY_NOTE)
            )

        if case_results:
            case_chunks = self._format_case_results(case_results[:MAX_PER_SOURCE])
            sections.append(CONTEXT_USER_CASE_CHUNKS_HEADER.format(case_chunks=case_chunks))
        else:
            sections.append(CONTEXT_USER_CASE_CHUNKS_HEADER.format(case_chunks=CONTEXT_EMPTY_NOTE))

        # 2. 知识图谱结果
        if kg_results:
            kg_text = self._format_kg_results(kg_results[:MAX_PER_SOURCE])
            sections.append(CONTEXT_KG_HEADER.format(kg_text=kg_text))
        else:
            sections.append(CONTEXT_KG_HEADER.format(kg_text=CONTEXT_EMPTY_NOTE))

        # 3. Toyhom 问答结果
        if toyhom_results:
            qa_text = self._format_toyhom_results(toyhom_results[:MAX_PER_SOURCE])
            sections.append(CONTEXT_TOYHOM_HEADER.format(qa_text=qa_text))
        else:
            sections.append(CONTEXT_TOYHOM_HEADER.format(qa_text=CONTEXT_EMPTY_NOTE))

        # --- 最终提示词 ---
        context = "\n".join(sections)
        query_block = self._format_query_block(query, query_info)
        return (
            system_prompt
            + confidence_note
            + context
            + query_block
            + "\n\n请根据以上资料回答用户的问题。"
        )

    # ------------------------------------------------------------------
    # 格式化辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format_kg_results(results: list[Dict]) -> str:
        """格式化知识图谱结果。"""
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            intent = r.get("intent", "")
            answer = r.get("answer", "")
            if len(answer) > MAX_RESULT_CHARS:
                answer = answer[:MAX_RESULT_CHARS] + "…"
            lines.append(f"[{i}] ({intent}) {answer}")
        return "\n".join(lines)

    @staticmethod
    def _format_toyhom_results(results: list[Dict]) -> str:
        """格式化 Toyhom 问答结果。"""
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            answer = r.get("answer", "")
            text = answer or title or ""
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + "…"
            department = r.get("department", "")
            prefix = f"[{i}] "
            if department:
                prefix += f"科室：{department} | "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines)

    @staticmethod
    def _format_case_results(results: list[Dict]) -> str:
        """格式化当前用户病例片段。"""
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            filename = r.get("filename", "")
            text = r.get("answer") or r.get("text") or ""
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + "…"
            prefix = f"[{i}] "
            if filename:
                prefix += f"文件：{filename} | "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines)

    @staticmethod
    def _format_query_block(query: str, query_info: Optional[Dict]) -> str:
        if not query_info:
            return f"\n\n## 用户当前问题\n{query}"
        normalized = query_info.get("normalized_query", query)
        reason = query_info.get("rewrite_reason", "")
        terms = "、".join(query_info.get("medical_terms") or [])
        return (
            "\n\n## 用户当前问题\n"
            f"原始问题：{query}\n"
            f"检索规范化问题：{normalized}\n"
            f"规范化说明：{reason}"
            + (f"\n医学术语：{terms}" if terms else "")
        )
