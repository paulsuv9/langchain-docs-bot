from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load vector store
embedding = HuggingFaceEmbeddings()
db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)

# Load local LLM
llm = pipeline("text-generation", model="microsoft/phi-2", device="cpu")

# Session memory
session_history = []


while True:
    query = input("\nAsk your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    # Retrieve top chunk
    docs = db.similarity_search(query, k=3)
    context = docs[0].page_content

    # Build prompt with memory
    memory_block = "\n".join([f"Q: {q}\nA: {a}" for q, a in session_history])
    prompt = (
        "You are a helpful assistant. Use the previous Q&A for context if relevant.\n\n"
        + memory_block
        + "\n\nCurrent excerpt:\n"
        + context
        + "\n\nQuestion: " + query + "\nAnswer:"
    )

    # Generate response
    response = llm(prompt, max_new_tokens=200)[0]['generated_text']
    print("\nLLM Response:\n", response)

    # Save to session memory
    session_history.append((query, response))

    # Log to file
    with open("logs/session_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n---\nQ: {query}\nA: {response}\n")

    
