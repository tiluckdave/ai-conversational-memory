# Natural Language Query Agent

> Submission of Take Home Assignment for the AI Resident role at Ema

This project demonstrates the creation of a Natural Language Query Agent to answer questions over a small set of lecture notes and a table of LLM architectures. The aim is to showcase the ability to work with data, implement techniques to solve tasks related to natural language processing, and provide conversational answers to simple queries.

## Approach

### Data Preprocessing

1. **Data Loading**:

   - **Lecture Notes**: Downloaded in markdown format to preserve formatting, including formulae, lists, and tables.
   - **LLM Architecture Table**: Downloaded from GitHub and converted to a CSV file to demonstrate handling different file formats.

2. **Data Splitting**:

   - **Markdown Files**: Used `TextLoader` to load markdown files while preserving formatting. Used `MarkdownHeaderTextSplitter` to split documents based on headers and `RecursiveCharacterTextSplitter` to further divide the documents into smaller chunks (chunk size of 1000).
   - **CSV File**: Used `CSVLoader` to load the CSV file, where each row is treated as a separate document.

3. **Metadata Enhancement**:
   - Added source file to the metadata of each chunk to track the file name.

### Embedding Generation and Vector Store

1. **Embeddings**:

   - Utilized `OpenAIEmbeddings` to generate embeddings for the document chunks.

2. **Vector Store**:
   - Created a Chroma vectorstore to store the embeddings, allowing for efficient retrieval of relevant documents.

### Retriever Setup

1. **Initialize Embeddings and Vector Store**:

   - Set up OpenAI embeddings and the Chroma vectorstore.

2. **Create Self-Query Retriever**:
   - Using a `SelfQueryRetriever` to enable the LLM to query the vectorstore using natural language. Provided additional metadata field information to guide the retriever on the structure of the documents.
   - Reason behind providing additional metadata is to provide direction to the retriever on how to retrieve the documents. For example, if the user asks a question about harms, the retriever should know that the harms are in the harms-1.md and harms-2.md files.

### Conversational Interface

1. **History-Aware Retriever**:

   - Created a system prompt which reformulates the user query in context of the chat history.
   - Leveraged `create_history_aware_retriever` to integrate the history-aware retriever with the LLM and the `SelfQueryRetriever`.

2. **QA Chain**:

   - Created a system prompt guiding the LLM to use retrieved documents to formulate answers.
   - Defined a QA prompt template with placeholders for system messages and chat history.
   - Used `create_stuff_documents_chain` to combine the LLM and the QA prompt template into a chain for answering questions.

3. **Conversational Chain**:

   - Used `create_retrieval_chain` to integrate the history-aware retriever and the QA chain.
   - Implemented `RunnableWithMessageHistory` to manage the conversational context and maintain the session history.

4. **Streaming Responses**:
   - Streams the response chunk-by-chunk to provide real-time feedback to the user.

## Setup and Usage

### Prerequisites

- Python 3.7+
- OpenAI API Key (Store in `.env` file similar to `.env.example`)
- Install required packages:

```bash
pip install -r requirements.txt
```

### Directory Structure

```plaintext
llm
└── data
    ├── lectures
    │   ├── capabilities.md
    │   ├── data.md
    │   ├── harms-1.md
    │   ├── harms-2.md
    │   ├── introduction.md
    │   └── legality.md
    └── table.csv
src
├── data_preprocessing.py
├── generate_embeddings.py
└── retriever.py
└── conversation.py
.env
.env.example
.gitignore
chat.py
main.py
README.md
requirements.txt
```

### Running the Project

**1. Data Preprocessing and Embedding Generation:**

```bash
python3 main.py
```

**2. Start the Conversational Interface:**

```bash
python3 chat.py
```

## Scalability

As the number of lectures and documents grow, we will have to implement for the documents and. We are currently using a simple vector store which store the embeddings in sqlite database.

We can use more advanced vector stores like FAISS which stores in memory and is faster. But it is not possible to store all the embeddings in the memory as the number of documents grow.

We can use a hybrid approach where we store the most relevant embeddings in memory and the rest in the databse.

Apart from this we can use more advanced retrievers which can be used to retrieve the embeddings faster as well as from multiple vectorstores at the same time.
