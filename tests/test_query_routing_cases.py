from medrag.retrieval.query_normalizer import QueryNormalizer
from medrag.retrieval.router import QueryRouter
from medrag.data import user_case_store
from medrag.infrastructure.storage import JsonStore


def test_query_normalizer_rewrites_colloquial_terms():
    result = QueryNormalizer().normalize("请问我嗓子疼还发烧怎么办？")

    assert result.original_query == "请问我嗓子疼还发烧怎么办？"
    assert "咽痛" in result.normalized_query
    assert "发热" in result.normalized_query
    assert result.medical_terms == ["咽痛", "发热"]


def test_router_adds_answer_style_and_case_flag():
    route = QueryRouter().route("我的检查报告尿酸高怎么办", use_llm=False)

    assert route["needs_case_context"] is True
    assert route["answer_style"] in {
        "test_report",
        "general_guidance",
        "disease_fact",
        "diet_guidance",
    }
    assert route["query_type"] in {
        "test_report",
        "general_medical_qa",
        "disease_fact",
        "diet",
    }


def test_user_case_store_isolates_users(tmp_path, monkeypatch):
    store_path = tmp_path / "user_cases.json"
    monkeypatch.setattr(user_case_store, "_case_store", JsonStore(str(store_path)))

    user_case_store.add_user_case(
        username="alice",
        filename="a.pdf",
        chunks=["空腹血糖 7.2 mmol/L，需要复查"],
        summary="检查/检验结果：空腹血糖 7.2 mmol/L",
    )
    user_case_store.add_user_case(
        username="bob",
        filename="b.pdf",
        chunks=["尿酸 560 umol/L"],
        summary="检查/检验结果：尿酸 560 umol/L",
    )

    retriever = user_case_store.UserCaseRetriever()
    alice_results = retriever.search("血糖复查", username="alice")
    bob_results = retriever.search("血糖复查", username="bob")

    assert alice_results
    assert alice_results[0]["username"] == "alice"
    assert bob_results == []


def test_milvus_insert_accepts_pk(monkeypatch):
    from medrag.vectors.milvus_client import MilvusClientWrapper

    inserted = []

    class FakeCollection:
        def insert(self, rows):
            inserted.extend(rows)

    wrapper = MilvusClientWrapper.__new__(MilvusClientWrapper)
    wrapper.collection = FakeCollection()
    wrapper.collection_name = "fake"
    wrapper.alias = "default"

    ok = wrapper.insert_batch(
        [{"pk": "case-1", "answer": "text", "text": "text"}],
        [[0.1, 0.2]],
    )

    assert ok is True
    assert inserted[0]["pk"] == "case-1"
