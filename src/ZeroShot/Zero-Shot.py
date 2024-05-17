"""Zero Shot on a LLM."""
import requests  # type: ignore
import json
import typing
import nltk  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from rouge_score.rouge_scorer import RougeScorer  # type: ignore
import argparse
import logging
from src.ZeroShot.ollama.ollama_gen import Ollama

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def generate_question_from_llm(
    claim: str, ollama: Ollama, example: typing.Dict, num_question: int
) -> typing.Dict:
    """Sends a request to a LLM to generate a question for a claim.

    Args:
        claim: The claim the model will generate a question for
        ollama: The Ollama object to generate from
        example: The example that the LLM should try to match for One-Shot.
    """

    QGEN_PROMPT_ALL = """Transform the given claim into questions to verify the \
        veracity of the given claim and find the potential correct facts from search engines.\
        The questions must be cover all aspects of the claim. \
        Generate exactly {num_questions} questions for the claim without making any references to the claim.\
        Here are some examples in different languages:
        Claim: "Kelvin Hopins was suspended from the Labor Party due to his membership in the Conservative Party."
        {{"Questions": ["Was Kelvin Hopins suspended from Labor Party?", "Why was Kelvin Hopins suspended from Labor Party?"]}}

        Generate the questions in the JSON format.

        Claim: "{claim}"
        """.strip()  # noqa E501
    prompt = QGEN_PROMPT_ALL.format(claim=claim, num_questions=num_question)

    # prompt = (
    #     "You transform a claim into a question. "
    #     + "The question can be understood without reading the claim. "
    #     + "You generate exactly one question per claim and nothing else."
    # )
    # user_input = (
    #     "Example:\n"
    #     + f"Claim: {example['input_text']}\n"
    #     + f"Question: {example['target_text']}\n\n"
    #     + "Your Response:\n"
    #     + f"Claim: {claim}\n"
    #     + "Question: "
    # )
    # print(prompt)
    data = ollama.generate(prompt)
    # print(data)
    try:
        return json.loads(data)
    except:
        return {"Questions": [""] * num_question}


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
    parser.add_argument(
        "--model",
        help=(
            "Choose between `llama2` and `mistral`"
        ),
        default="mistral",
    )

    args = parser.parse_args()
    model = str(vars(args)["model"]).strip()
    # print(model)
    config_path = "src/ZeroShot/ollama/" + model + ".yaml"
    # print(config_path)
    dataset_path = str(vars(args)["dataset"]).strip()
    ollama = Ollama(config_path)
    scores = []

    data = load_from_json("data/" + dataset_path + "/test.json")
    example = {
        "input_text": "African American unemployment is the lowest ever recorded in our country. The Hispanic unemployment rate dropped a full point in the last year and is close to the lowest in recorded history. Dems did nothing for you but get your vote!",
        "target_text": "Are unemployment rates for African Americans and Hispanics low today?",
    }
    freq = {}
    for example in data:
        if example["input_text"] in freq:
            freq[example["input_text"]].append(example["target_text"])
        else:
            freq[example["input_text"]] = [example["target_text"]]

    for inp,target in freq.items():
        claim = inp
        gen_question = generate_question_from_llm(
            claim=claim, ollama=ollama, example=example, num_question=len(target)
        )
        gen_question = gen_question[list(gen_question.keys())[0]]
        print("before",gen_question)

        if not type(gen_question) == list:
            gen_question = [""] * len(target)
        elif len(gen_question) < len(target):
            gen_question = [""]*len(target)
        else:
            for z in gen_question:
                if not type(z) == str:
                    gen_question = [""] * len(target)
                    break
        gen_question = gen_question[:len(target)]
        print("after",target, gen_question)
        for l in range(len(target)):
            rouge_score = rouge(target[l], gen_question[l])
            bleu_score = bleu(target[l], gen_question[l])

            scores.append(
                {
                    "rouge 1": rouge_score["rouge1"],
                    "rouge L": rouge_score["rougeL"],
                    "bleu": bleu_score,
                    "claim": claim,
                    "target_text": target[l],
                    "generated": gen_question[l],
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

    with open(
        "src/ZeroShot/results/" + dataset_path + "_" + model + ".json",
        "w",
    ) as file:
        json.dump(scores, file, indent=4)
