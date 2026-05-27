"""NER 模型加载器，从 MedicalChatService._load_kg_retriever 提取而来。"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import torch
from transformers import BertTokenizer

from medrag.ner import model as zwk
from medrag.retrieval.kg_retriever import KGRetriever

logger = logging.getLogger(__name__)


def load_ner_model(
    project_root: str | Path,
    checkpoint: str = "best_roberta_rnn_model_ent_aug.pt",
) -> KGRetriever:
    """加载 NER 流水线（RoBERTa-RNN + 规则 + TF-IDF），返回可用的 KGRetriever。"""
    root = Path(project_root)

    tag2idx_path = root / "tmp_data" / "tag2idx.npy"
    with open(tag2idx_path, "rb") as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)

    rule = zwk.rule_find()
    tfidf_r = zwk.tfidf_alignment()

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    bert = zwk.Bert_Model(
        "hfl/chinese-roberta-wwm-ext",
        hidden_size=128,
        tag_num=len(tag2idx),
        bi=True,
    )

    checkpoint_path = root / "model" / checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    bert = bert.to(device)
    bert.eval()

    logger.info("NER model loaded (checkpoint=%s, device=%s)", checkpoint, device)

    return KGRetriever(
        bert_model=bert,
        bert_tokenizer=tokenizer,
        rule=rule,
        tfidf_r=tfidf_r,
        device=device,
        idx2tag=idx2tag,
    )
