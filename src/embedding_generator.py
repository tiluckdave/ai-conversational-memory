import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.data_preprocessing import load_lecture_notes, load_architecture_table

load_dotenv()

def generate_embeddings(persist_directory='data/chroma/'):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings()

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    notes = load_lecture_notes("data/lectures/")
    table = load_architecture_table("data/table.csv")

    docs = notes + table

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectordb

if __name__ == "__main__":
    vectordb = generate_embeddings()
    print(vectordb._collection.count()) 
