"""Using One Shot on a LLM."""
import requests  # type: ignore
import json
import typing
import nltk  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from rouge_score.rouge_scorer import RougeScorer  # type: ignore
import argparse
import logging
from src.OneShot.ollama.ollama_gen import Ollama
from openai_request import OpenAIUtils

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def generate_question_from_llm(
    claim: str, model: Ollama | OpenAIUtils, num_question: int
) -> typing.Dict:
    """Sends a request to a LLM to generate a question for a claim.

    Args:
        claim: The claim the model will generate a question for
        model: The Ollama object to generate from
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
    data = model.generate(prompt)
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

def write_results(filename: str):
    """Write JSON results to file `filename`

    Args:
        filename: the name of the file to write to
    """
    with open(filename, "w") as file:
        json.dump(scores, file, indent=4)

def arguments() -> typing.Dict:
    """Parses code's inputs and returns the parser

    Returns:
        A dictionary of arguments
    """
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

    return parser.parse_args()

def dictionary_cq(raw_data: typing.Dict) -> typing.Dict:
    """Reformates the input dictionary. Ensures all claims are paired
    with their question set.


    Args:
        raw_data (typing.Dict): Raw data from a dataset's test file

    Returns:
        typing.Dict: Reformat to keys = claims, values = question-set.
    """
    freq = {}
    for example in raw_data:
        if example["input_text"] in freq:
            freq[example["input_text"]].append(example["target_text"])
        else:
            freq[example["input_text"]] = [example["target_text"]]
    return freq

if __name__ == "__main__":
    args = arguments()
    model = str(vars(args)["model"]).strip()
    config_path = "src/ZeroShot/ollama/" + model + ".yaml"
    dataset_path = str(vars(args)["dataset"]).strip()
    if args["model"] == "gpt4o":
        model = OpenAIUtils()
    else:
        model = Ollama(config_path)
    scores = []

    data = load_from_json("data/" + dataset_path + "/test.json")
    freq = dictionary_cq(data)

    for claim,target in freq.items():
        gen_question = generate_question_from_llm(
            claim=claim, model=model, num_question=len(target)
        )
        gen_question = gen_question[list(gen_question.keys())[0]]

        #Parse dictionary for questions
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
    write_results("src/ZeroShot/results/" + dataset_path + "_" + model + ".json")
