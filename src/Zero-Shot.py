"""Zero Shot on a LLM."""
import requests  # type: ignore
import json
import typing
import nltk  # type: ignore
from rouge_score import rouge_scorer  # type: ignore


def generate_question_from_llm(
    claim: str, url: str, session: requests.Session
) -> typing.Dict:
    """Sends a request to a LLM to generate a question for a claim.

    Args:
        claim: The claim the model will generate a question for
        url: The URL to a LLM server which takes in a POST request
            The arguments of the POST request is `prompt` and `n_predict`
            for the maximum tokens to generate.
        session: The requests session we are sending our request in
    """
    headers = {"Content-Type": "application/json"}
    prompt = (
        "Generate one question for the claim: "
        + claim
        + "Do not reply using a complete sentence, \
        and only give the answer in one sentence."
    )
    data = {
        "prompt": prompt,
        "n_predict": 128,
    }
    response = session.post(url=url, json=data, headers=headers)
    return response.json()


def tokenize(text: str) -> list:
    """Tokenize an input string.

    Args:
        text: string to be tokenized

    Returns:
        tokenized list
    """
    return nltk.tokenize.word_tokenize(text)


def bleu(reference: str, hypothesis: str) -> float:
    """Calculate the Bleu score of reference, hypothesis pair.

    Args:
        reference: The target text that the model is trying to generate
        hypothesis: The text that the model generated.

    Returns:
        Bleu score
    """
    reference_tokenize = tokenize(reference)
    hypothesis_tokenize = tokenize(hypothesis)
    return nltk.translate.bleu_score.sentence_bleu(
        [reference_tokenize], hypothesis_tokenize
    )


def rouge(reference: str, hypothesis: str) -> typing.Dict:
    """Calculate the Rouge score of reference, hypothesis pair.

    Args:
        reference: The target text that the model is trying to generate
        hypothesis: The text that the model generated.

    Returns:
        Different versions of Rouge scores.
    """
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    return rouge.score(reference, hypothesis)


def load_from_json(path: str) -> typing.Dict:
    """Loads data from a JSON file.

    Args:
        path: the path to the JSON file

    Returns:
        A dictionary of the JSON file
    """
    with open(path, "r") as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    data = load_from_json("questgen/data/gpt_generated/test.json")
    scores = []
    s = requests.session()
    url = ""  # omitted

    for d in data[0:1]:
        claim = d["input_text"]
        gen_question = generate_question_from_llm(
            claim=claim, url=url, session=s
        )["text"]
        target = d["target_text"]
        rouge_score = rouge(target, gen_question)
        bleu_score = bleu(target, gen_question)
        scores.append(
            {
                "rouge 1": rouge_score["rouge1"],
                "rouge L": rouge_score["rougeL"],
                "bleu": bleu_score,
                "claim": claim,
                "target_text": target,
                "generated": gen_question,
            }
        )

    scores.append(
        {
            "average": True,
            "rouge 1": {
                "precision": sum([s["rouge1"].precision for s in scores])
                / len(scores),
                "recall": sum([s["rouge1"].precision for s in scores])
                / len(scores),
                "fmeasure": sum([s["rouge1"].precision for s in scores])
                / len(scores),
            },
            "rouge L": {
                "precision": sum([s["rougeL"].precision for s in scores])
                / len(scores),
                "recall": sum([s["rougeL"].precision for s in scores])
                / len(scores),
                "fmeasure": sum([s["rougeL"].precision for s in scores])
                / len(scores),
            },
            "bleu": sum([s["bleu"] for s in scores]) / len(scores),
        }
    )

    with open("questgen/src/ZeroShot/gpt_generated.json", "w") as file:
        json.dump(scores, file)
