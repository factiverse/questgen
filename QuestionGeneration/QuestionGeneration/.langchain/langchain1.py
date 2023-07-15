from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import dotenv



def init_langchain():
    global retriever
    dotenv.load_dotenv(".env")
    loader = TextLoader('train_claims.txt', encoding='utf8')

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    #What we run to get query

def run_query(query):
    global retriever
    docs = retriever.get_relevant_documents(query)
    return docs


def docs_to_list(docs):
    final_list = []
    for i in range(len(docs)):
        final_list.append(docs[i].page_content)
    return final_list

init_langchain()
docs = run_query("Says Facebook shut down a Chick-Fil-A Appreciation Day.")
print(docs_to_list(docs=docs))





