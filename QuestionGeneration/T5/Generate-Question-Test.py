import transformers
from datasets import load_dataset
from evaluate import load
import datasets
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import nltk
import numpy as np
# import huggingface_hub

model_checkpoint = "/home/ritvik/QuestionGeneration/T5/wandb/run-20230709_044116-c1924a9g/files/model/checkpoint-23500"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    """_summary_

    Args:
        examples (_type_): _description_

    Returns:
        _type_: _description_
    """
    inputs = [pre + inp for inp,pre in zip(examples["input_text"],examples["prefix"]) ]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["target_text"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

print("type something")
while True:
    inp = input()
    if inp == 'q':
        break
    input_ids = tokenizer(inp, return_tensors="pt")
    print(input_ids['input_ids'])
    outputs = model.generate(input_ids['input_ids'])
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# model.generate()
#What is the origin of the claim that the earth is flat?
#What is the origin of the claim that the earth is flat?
#