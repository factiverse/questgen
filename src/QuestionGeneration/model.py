import json
import openai
from nltk.translate.bleu_score import sentence_bleu
import nltk
from rouge import Rouge 
import csv
import json
import random
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import dotenv

dotenv.load_dotenv(".env")

def init_langchain():
    global retriever
    loader = TextLoader('train_claims.txt', encoding='utf8')
    dotenv.load_dotenv(".env")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print('requesting openai auth')
    embeddings = OpenAIEmbeddings(openai_api_key='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')

    db = FAISS.from_documents(texts, embeddings)
    print('finished requesting openai auth')

    retriever = db.as_retriever(search_kwargs={"k": 4})
    #What we run to get query

def run_query(query):
    global retriever
    docs = retriever.get_relevant_documents(query)
    # print(docs, type(docs))
    return docs


def docs_to_list(docs):
    final_list = []
    # print(docs, type(docs))
    for i in range(len(docs)):
        final_list.append(docs[i].page_content)
    return final_list

# init_langchain()
# docs = run_query("Says Facebook shut down a Chick-Fil-A Appreciation Day.")
# print(docs_to_list(docs=docs))

def read_csv_cq(file_name):
    claims = []
    questions = []
    with open(file_name, newline='\n', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            claims.append(row['claim_reviewed'])
            questions.append(row['question'])

    return claims, questions
    
def list_to_dict(claims, questions):
    claim2q = {}
    for c in claims:
        claim2q[c] = []

    for c,q in zip(claims, questions):
        claim2q[c].append(q)

    return claim2q

def store_info(**kwargs):
    store = {}
    for k,v in kwargs.items():
        store[k] = v
    return store
    
def write_info(file_name, store):
    with open(file_name, 'w') as file:
        json.dump(store, file, indent=4)

def init():
    global deployment_name
    init_langchain()

    print("----------------------------------------------------------------------------")
    # # openai.api_key = dotenv.get_key(".env",'MODEL_API_KEY')
    # # openai.api_base = dotenv.get_key(".env",'MODEL_API_BASE') # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    # # openai.api_type = dotenv.get_key(".env",'MODEL_API_TYPE')
    # # openai.api_version = dotenv.get_key(".env",'MODEL_API_VERSION') # this may change in the future
    openai.api_key = "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"
    # # openai.api_key = dotenv.get_key(".env",'OPENAI_API_KEY')
    # openai.api_base = "https://openaifactiverse.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    # openai.api_type = "azure"
    # openai.api_version = "2022-12-01" # this may change in the future
    # openai.api_key = "73d692b52999433294f968acbb5cec74"
    # openai.api_base =  "https://openaifactiverse.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    # openai.api_type = 'azure'
    # openai.api_version = '2022-12-01' # this may change in the future
    print("*********************************************************************************")
    deployment_name='text-davinci-003' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

# Send a completion call to generate an answer

def ask_model(start_phrase):
    global deployment_name
    deployment_name='text-davinci-003'
    response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=100)
    text = response['choices'][0]['text'].replace(' .', '.').strip()
    return text

def prompt_construct(claim, file_train, num_questions):
        demo_claim = docs_to_list(run_query(claim))[0]
        prompt = f"If the claim \"{demo_claim}\" can be answered by the questions: "
        questions = file_train[demo_claim.strip('\'').strip('\"')]
        for q in range(len(questions)):
            prompt = prompt + questions[q]
        prompt=prompt+" "

        prompt = prompt + "Can you generate " + str(num_questions) + " questions that answer the claim, \""+claim+"\"? "
        return prompt


#TODO: ask the right number of questions. Currently, questions are being asked

def run(MAX):
    claims, questions = read_csv_cq("test.csv")
    file_test = list_to_dict(claims, questions)    

    claims, questions = read_csv_cq("train.csv")
    file_train = list_to_dict(claims,questions)
    records = []
    for k,v,i in zip(file_test.keys(), file_test.values(), range(len(file_test.keys()))):
        # rand = random.randint(0,len(claims))
        store = {}
        reference = []
        claim = k
        store["claim"] = claim

        prompt = prompt_construct(k,file_train, len(v))
        for q in range(len(v)):
            reference.append(v[q])
        
        text = ask_model(prompt)
        hypothesis = text.split("\n")
        print("answer:",text, "\nend statement", len(text.split("\n")))
        print(k,v,"\n",prompt)
        bleu = bleu_score(reference, hypothesis)
        rouge = rouge_score(reference, hypothesis)
        records.append(store_info(claim=claim, reference=reference, hypothesis=hypothesis, bleu=bleu, rouge=rouge, prompt=prompt))
        if i == MAX:
            break
    write_info("store.json", records)

def rouge_score(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    rouge_l = scores[0]['rouge-l']
    return rouge_l

def bleu_score(reference, hypothesis):
    bleu = []
    reference_tokenize = []
    for ref in reference:
        reference_tokenize.append(nltk.word_tokenize(ref))
    for i in hypothesis:
        hypothesis_tokenize = nltk.word_tokenize(i)
        bleu.append(sentence_bleu(reference_tokenize, hypothesis_tokenize, weights=(0.25, 0.25, 0.25, 0.25)))
    return bleu

init()
run(1)


# claims, questions = read_csv_cq("train.csv")
# file_test = list_to_dict(claims, questions)   
# # print(claims)
# print(file_test["""A photograph shows a baby great white shark.\"""".strip('\'').strip('\"')])
# with open("store-claims", 'w') as file:
#     file.write(str(claims))

# with open("store-claims-1", 'w') as file:
#     file.write(str(file_test.keys()))
#FAILED FOR 'A photograph shows a baby great white shark.'


















#Homework = Feed everything into the model and store rouge-l and bleu scores, all questions, claims and model's results
# Also compare its results to the https://docs.cohere.ai/reference/about model

#Read training data
#Select random training data to go with testing data. Feed in Training claim, training questions, testing claim, and ask it to generate testing questions.
#Compare the testing questions to the real testing questions.

#Use dense search to encode each sentence as a vector. Then use k-nearest in order to find k claims that can be used to prompt the model to question our claim.
#https://python.langchain.com/en/latest/modules/indexes/vectorstores.html




#Simple Transformers: https://simpletransformers.ai/docs/t5-specifics/

#write code t