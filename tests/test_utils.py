import pytest

from src.util import compute_bert_score, compute_meteor


@pytest.fixture
def bleu_rouge_score():
    return {}


def test_compute_meteor(bleu_rouge_score):
    decoded_preds = ["Earth is flat"]
    decoded_labels = ["Earth is not flat"]
    compute_meteor(decoded_preds, decoded_labels, bleu_rouge_score)

    assert "meteor" in bleu_rouge_score
    assert isinstance(bleu_rouge_score["meteor"], float)
    assert bleu_rouge_score["meteor"] > 0.5

    decoded_preds = ["What is your name?"]
    decoded_labels = ["What is your name?"]
    compute_meteor(decoded_preds, decoded_labels, bleu_rouge_score)

    assert "meteor" in bleu_rouge_score
    assert isinstance(bleu_rouge_score["meteor"], float)
    assert bleu_rouge_score["meteor"] == 0.996


def test_compute_bert_score(bleu_rouge_score):
    decoded_preds = ["Earth is flat"]
    decoded_labels = ["Earth is not flat"]
    compute_bert_score(decoded_preds, decoded_labels, bleu_rouge_score)

    assert "bert_score_f1" in bleu_rouge_score
    assert "bert_score_precision" in bleu_rouge_score
    assert "bert_score_recall" in bleu_rouge_score
    assert isinstance(bleu_rouge_score["bert_score_f1"], float)
    assert bleu_rouge_score["bert_score_f1"] > 0
    assert isinstance(bleu_rouge_score["bert_score_precision"], float)
    assert bleu_rouge_score["bert_score_precision"] > 0
    assert isinstance(bleu_rouge_score["bert_score_recall"], float)
    assert bleu_rouge_score["bert_score_recall"] > 0

    decoded_preds = ["What is your name?"]
    decoded_labels = ["What is your name?"]
    compute_bert_score(decoded_preds, decoded_labels, bleu_rouge_score)

    assert "bert_score_f1" in bleu_rouge_score
    assert isinstance(bleu_rouge_score["bert_score_f1"], float)
    assert bleu_rouge_score["bert_score_f1"] == 1.0
