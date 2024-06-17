import os
import openai
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

load_dotenv()

def initialize_embeddings():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return OpenAIEmbeddings()

def initialize_vector_db(persist_directory='data/chroma/'):
    embeddings = initialize_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def create_self_query_retriever(llm, vectordb):
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of `data/lectures/introduction.md`, `data/lectures/data.md`, `data/lectures/capabilities.md`, `data/lectures/legality.md`, data/lectures/harms-1.md`, `data/lectures/harms-2.md`, `data/table.csv`",
            type="string",
        ),
    ]

    return SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        "Lecture Notes and Architecture",
        metadata_field_info,
        verbose=False
    )

def initialize_llm():
    return ChatOpenAI(temperature=0, model="gpt-4")

def create_retriever(persist_directory='data/chroma/'):
    vectordb = initialize_vector_db(persist_directory)
    llm = initialize_llm()
    return create_self_query_retriever(llm, vectordb)
