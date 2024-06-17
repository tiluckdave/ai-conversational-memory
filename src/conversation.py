from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.retriever import create_retriever, initialize_llm
import logging

logging.getLogger().setLevel(logging.ERROR)

def create_history_aware_retriever_chain(llm, retriever):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

def create_qa_chain(llm):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return create_stuff_documents_chain(llm, qa_prompt)

def create_rag_chain(persist_directory='data/chroma/'):
    retriever = create_retriever(persist_directory)
    llm = initialize_llm()
    history_aware_retriever = create_history_aware_retriever_chain(llm, retriever)
    qa_chain = create_qa_chain(llm)
    return create_retrieval_chain(history_aware_retriever, qa_chain)

def initialize_conversational_rag_chain(persist_directory='data/chroma/'):
    rag_chain = create_rag_chain(persist_directory)
    
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def run_query(conversational_rag_chain, query, session_id):
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    return response["answer"]

def stream_query(conversational_rag_chain, query, session_id):
    return conversational_rag_chain.stream(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
