"""Phase 2 单元测试：LLM 依赖注入。"""

from unittest.mock import MagicMock

import pytest

from medrag.llm import LLMProvider, get_llm_provider


class TestLLMProvider:
    def test_bundles_client_and_model(self):
        client = MagicMock()
        lp = LLMProvider(name="deepseek", client=client, default_model="deepseek-chat")
        assert lp.name == "deepseek"
        assert lp.client is client
        assert lp.default_model == "deepseek-chat"

    def test_is_frozen_dataclass(self):
        client = MagicMock()
        lp = LLMProvider(name="deepseek", client=client, default_model="deepseek-chat")
        with pytest.raises(Exception):
            lp.name = "other"  # type: ignore[misc]


class TestGetLLMProvider:
    def test_returns_correct_provider_name(self):
        lp = get_llm_provider()
        assert lp.name in ("deepseek", "zhipuai", "ollama")

    def test_returns_openai_compatible_client(self):
        lp = get_llm_provider()
        assert hasattr(lp.client, "chat")
        assert hasattr(lp.client.chat, "completions")

    def test_caches_client(self):
        lp1 = get_llm_provider()
        lp2 = get_llm_provider()
        assert lp1.client is lp2.client

    def test_raises_for_unknown_provider(self):
        with pytest.raises(ValueError, match="不支持的 LLM_PROVIDER"):
            get_llm_provider("unknown_provider_xyz")


class TestAnswerGeneratorInjection:
    def test_accepts_injected_provider(self):
        from medrag.rag.answer_generator import AnswerGenerator

        client = MagicMock()
        lp = LLMProvider(name="test", client=client, default_model="test-model")
        ag = AnswerGenerator(llm_provider=lp)
        assert ag._client is client
        assert ag._model == "test-model"

    def test_defaults_to_global_provider(self):
        from medrag.rag.answer_generator import AnswerGenerator

        ag = AnswerGenerator()
        default_lp = get_llm_provider()
        assert ag._client is default_lp.client
        assert ag._model == default_lp.default_model


class TestKGRetrieverInjection:
    def test_accepts_llm_client_parameter(self):
        from medrag.retrieval.kg_retriever import KGRetriever
        import inspect

        sig = inspect.signature(KGRetriever.__init__)
        assert "llm_client" in sig.parameters

    def test_llm_client_in_signature_order(self):
        from medrag.retrieval.kg_retriever import KGRetriever
        import inspect

        sig = inspect.signature(KGRetriever.__init__)
        params = list(sig.parameters.keys())
        assert "llm_client" in params
