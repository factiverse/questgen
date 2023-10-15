"""Zero Shot on a LLM."""
import requests  # type: ignore
import json
import typing
import nltk  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from rouge_score.rouge_scorer import RougeScorer  # type: ignore
import argparse


def generate_question_from_llm(
    claim: str, url: str, session: requests.Session, example: typing.Dict
) -> typing.Dict:
    """Sends a request to a LLM to generate a question for a claim.

    Args:
        claim: The claim the model will generate a question for
        url: The URL to a LLM server which takes in a POST request
            The arguments of the POST request is `prompt` and `n_predict`
            for the maximum tokens to generate.
        session: The requests session we are sending our request in
        example: The example that the LLM should try to match for One-Shot.
    """
    headers = {"Content-Type": "application/json"}

    prompt = (
        "You transform a claim into a question. "
        + "The question can be understood without reading the claim. "
        + "You generate exactly one question per claim and nothing else."
    )
    user_input = (
        "Example:\n"
        + f"Claim: {example['input_text']}\n"
        + f"Question: {example['target_text']}\n\n"
        + "Your Response:\n"
        + f"Claim: {claim}\n"
        + "Question: "
    )

    # print(prompt+"\n\nUser: "+user_input+"\n\nllama:")
    data = {
        "prompt": prompt + "\n\n" + user_input,
        "n_predict": 400,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.1,
        "repeat_penalty": 1.18,
        "repeat_last_n": 256,
        "seed": 42,
        "frequency_penalty": 0,
        # "grammar": 0,
        "mirostat": 0,
        "mirostat_eta": 0.1,
        "mirostat_tau": 5,
        "presence_penalty": 0,
        "stop": ["</s>", "llama:", "User:"],
        # "stream": "true",
        "tfs_z": 1,
        "typical_p": 1,
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
    tokenizer = RegexpTokenizer(r"\w+")
    tokenizer.tokenize("Eighty-seven miles to go, yet.  Onward!")
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
    rouge = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help=(
            "path to dataset. Only include the name of the folder."
            + "Ex. data/{Your Input}/test.json should exist"
        ),
        default="sample",
    )
    args = parser.parse_args()
    dataset_path = str(vars(args)["dataset"]).strip()

    data = load_from_json("data/" + dataset_path + "/test.json")
    scores = []
    s = requests.session()
    url = "http://llama.factiverse.ai:8080/completion"

    for d in data[1:]:
        claim = d["input_text"]
        gen_question = generate_question_from_llm(
            claim=claim, url=url, session=s, example=data[0]
        )[
            "content"
        ]  # Could be a different key
        target = d["target_text"]
        rouge_score = rouge(target, gen_question)
        bleu_score = bleu(target, gen_question)
        # print(rouge_score)
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
                "precision": sum([s["rouge 1"].precision for s in scores])
                / len(scores),
                "recall": sum([s["rouge 1"].precision for s in scores])
                / len(scores),
                "fmeasure": sum([s["rouge 1"].precision for s in scores])
                / len(scores),
            },
            "rouge L": {
                "precision": sum([s["rouge L"].precision for s in scores])
                / len(scores),
                "recall": sum([s["rouge L"].precision for s in scores])
                / len(scores),
                "fmeasure": sum([s["rouge L"].precision for s in scores])
                / len(scores),
            },
            "bleu": sum([s["bleu"] for s in scores]) / len(scores),
        }
    )

    with open("src/ZeroShot/" + dataset_path + ".json", "w") as file:
        json.dump(scores, file, indent=4)
