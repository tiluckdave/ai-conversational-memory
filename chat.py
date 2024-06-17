import sys
from src.conversation import initialize_conversational_rag_chain, run_query, stream_query

if __name__ == "__main__":
    try:
        conversational_rag_chain = initialize_conversational_rag_chain()
    except Exception as e:
        print(f"Error initializing conversational RAG chain: {e}")
        exit(1)
        
    print("Commands")
    print("Type 'exit' or 'quit' to exit the chat")

    session_id = "cli_session"
    print("Enter your query: ", end="", flush=True)
    for line in sys.stdin:
        query = line.strip()
        if query.lower() in ["exit", "quit"]:
            break
        output = {}
        print("Answer: ", end="", flush=True)
        for chunk in stream_query(conversational_rag_chain, query, session_id):
            for key in chunk:
                if key not in output:
                    output[key] = chunk[key]
                else:
                    output[key] += chunk[key]
                if key == "answer":
                    print(chunk[key], end="", flush=True)
        
        print("\n\nSources: ", flush=True)
        for source in output['context']:
            print(source.metadata['source'], end=", ", flush=True)
            if source.metadata['source'] == 'data/table.csv':
                print("Row Number: ", end="", flush=True)
                print(source.metadata['row'])
        
        print("\n\nEnter your query: ", end="", flush=True)
