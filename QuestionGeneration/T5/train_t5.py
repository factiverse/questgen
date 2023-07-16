#!/usr/bin/env python
from simpletransformers.t5 import T5Model, T5Args
import json
import pandas as pd
from T5_functions import load, load_data, sort_data, set_seed
import wandb
import nltk
from rouge import Rouge 
import os
from transformers import T5Tokenizer,EvalPrediction
import torch 
import sacrebleu


tokenizer = T5Tokenizer.from_pretrained('t5-small')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
nltk.download("punkt")
set_seed()
data = load_data("/home/ritvik/QuestionGeneration/T5/data/all_en.json")
train, test = sort_data(data, data_fraction=0.0001)

model_args = T5Args()
model_args.num_train_epochs = 1
model_args.train_batch_size = 16
model_args.overwrite_output_dir = True
model_args.max_seq_length = 1024
model_args.reprocess_input_data = True
model_args.evaluate_during_training = True
model_args.manual_seed = 42
model_args.use_multiprocessing = True
model_args.eval_batch_size = 16
model_args.wandb_project = "Simple Sweep"
model_args.evaluate_during_training_steps = 100
# model_args.dropout = 0.2


def bleu_score(labels, preds):
    sum = []
    for label, pred in zip(labels, preds):
        ref = nltk.word_tokenize(label)
        hyp = nltk.word_tokenize(pred)
        sum.append(nltk.translate.bleu_score.sentence_bleu([ref], hyp))
    return sum

def rouge_score(labels, preds):
    rouge = Rouge()
    final_score = []
    for pred,label in zip(preds,labels):
        try:
            score = rouge.get_scores(pred, label)
            final_score.append(score[0]['rouge-l']['f'])
        except ValueError:
            final_score.append(0)
    return final_score

# def compute_metrics(labels_preds):# preds, *, normalize=True, sample_weight=None
#     preds, labels = labels_preds
#     rouge_vals = rouge_score(labels, preds)
#     bleu_vals = bleu_score(labels, preds)
#     return [(rouge_val+bleu_val)/2 for rouge_val,bleu_val in zip(rouge_vals,bleu_vals)]


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    decoded_preds = tokenizer.batch_decode(torch.argmax(preds, dim=2), skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = p.label_ids
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # SacreBLEU expects detokenized text
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    return {"bleu_score": bleu.score}

sweep_config = { #Tune: Learning Rate, Batch size, Dropout, weight decay
    "method": "bayes",  # grid, random
    "metric": {"name": "combined_metric", "goal": "maximize"},
    "parameters": {
        "weight_decay": {"values": [0.1, 0.01, 0.001]},
        "learning_rate": {"min": 4e-5, "max": 4e-3},
        "dropout": {"values": [0.1,0.2,0.3]}
    },

}

sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")

# wandb_tags = ["question generation"] if wandb_tags is None else wandb_tags
# wandb.config.update(model_args)

def train_model():
    wandb.init()
    model = T5Model(model_type="t5", model_name="t5-small", args=model_args, use_cuda=True, sweep_config=wandb.config, compute_metrics=compute_metrics) #, args={'wandb_project': 'project-name'}
    # train,test = load("./data/1/train_1.json", "./data/1/test_1.json")
    model.train_model(train,eval_data=test)
    model.eval_model(test)
 
    #using bleu and rouge scores here:
    input_text = test['input_text'].values.tolist()
    target_text = test['target_text'].values.tolist()
    result = model.predict(['generate question: ' + it for it in input_text])
    combined_metric_results = compute_metrics([target_text, result])
    final_score = 0
    for i in combined_metric_results:
        final_score += i
    final_score /= len(combined_metric_results)
    wandb.log({"combined_metric": final_score})
    wandb.join()

train_model()
# wandb.agent(sweep_id, train_model)

# with open("model_results_t5.txt", "w") as file:
#     file.write(json.dumps(result, indent=4))


# TODO: Use Rouge and Bleu scores to see how well the model is doing
# Use different models (currently using t5-base, there are others like )
# https://huggingface.co/google/flan-t5-small
