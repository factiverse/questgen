from simpletransformers.t5 import T5Model, T5Args

# from simpletransformers.question_answering import QuestionAnsweringModel

from T5_functions import load_data, sort_data, set_seed
import json


# model = T5Model(
#     "t5", "t5-base"
# )

set_seed()

model = T5Model("t5", "/home/ritvik/QuestionGeneration/T5/outputs/")

data = load_data("/home/ritvik/QuestionGeneration/T5/data/all_en.json")
train, test = sort_data(data)
results = model.eval_model(test)
# print(results)
with open("/home/ritvik/QuestionGeneration/T5/T5_model_results.txt", "w") as file:
    json.dump(results, fp=file, indent=4)

to_predict = ["generate question: Prime Minister of India is Modi"]

prediction = model.predict(to_predict)
print(prediction)
