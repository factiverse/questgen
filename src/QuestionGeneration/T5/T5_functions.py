import json
import pandas as pd
import random
import numpy as np
import torch
import random

def load_data(file_name):  # Just use load() to load from train.json and test.json
    # new_data = []
    with open(file_name, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    new_data = df.explode("gpt3_queries")
    # for i in range(len(new_data["gpt3_queries"])):
    #     if type(new_data["gpt3_queries"][i]) != type("hello"):
    #         print(new_data["claim"][i], i, i%3, new_data["gpt3_queries"][i])
    return new_data


def sort_data(
    df, split_decimal=0.9, data_fraction=1
):  # Just use load() to load from train.json and test.json
    def train_test_split(data, train_num, end):
        shuffled = data.sample(frac=1, random_state=1)
        train = shuffled[0:train_num]
        test = shuffled[train_num:end]
        return train, test

    df.rename(
        columns={"gpt3_queries": "target_text", "claim": "input_text"}, inplace=True
    )
    df["prefix"] = "generate question"
    train, test = train_test_split(
        df,
        int(split_decimal * df.__len__() * data_fraction),
        int(data_fraction * df.__len__()),
    )
    # train = train.drop("level_0")
    # test = test.reset_index()
    # print(train.columns)
    train = train.dropna()
    test = test.dropna()
    return train, test


def load(train_file_name, test_file_name):
    test = pd.read_json(test_file_name)
    train = pd.read_json(train_file_name)
    return train, test


def write_data(train, test, train_file_name, test_file_name):
    train.to_json(train_file_name, indent=4, orient="record")
    test.to_json(test_file_name, indent=4, orient="record")

def rewrite_data(all_data_file_name, new_data_file_name): #train, val, test: 64, 16, 20
    # text=[]
    with open(all_data_file_name, 'r') as file:
        text = json.load(file) #claim, gpt3_queries
    new_data = []
    prefixes = {1:"generate question", 2:"rewrite the claim", 0:"generate a query"}
    for i in range(len(text)):
        for j in range(len(text[i]['gpt3_queries'])):
            new_data.append({"input_text":text[i]['claim'], "target_text":text[i]['gpt3_queries'][j], "prefix":prefixes[j%3]})
    for i in range(20):
        random.shuffle(new_data)
    len_data = len(new_data)
    train_data_len = int(0.8*len_data)
    val_data_len = int(0.2*len_data)
    # test_data_len = 0.2*len_data # assume that it will be the rest
    train_data = new_data[:train_data_len]
    val_data = new_data[train_data_len:train_data_len+val_data_len]
    # test_data = new_data[train_data_len+val_data_len:]

    # print(text[0]['gpt3_queries'][0])
    with open(new_data_file_name+'train.json', 'w') as file:
        json.dump(train_data,file, indent=4)
    with open(new_data_file_name+'test.json', 'w') as file:
        json.dump(val_data,file, indent=4)
    # with open(new_data_file_name+'val.json', 'w') as file:
    #     json.dump(test_data,file, indent=4)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


rewrite_data("/home/ritvik/QuestionGeneration/T5/data/all_en.json", "/home/ritvik/QuestionGeneration/T5/data/unique-prefix/")

# data = load_data("./data/all_en.json")
# train,test = sort_data(data)
# print(train.head())
# print(test.head())
# write_data(train,test,"./data/1/train_1.json", "./data/1/test_1.json")
