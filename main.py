from src.embedding_generator import generate_embeddings

if __name__ == "__main__":
    vectordb = generate_embeddings()
    print(vectordb._collection.count()) 
