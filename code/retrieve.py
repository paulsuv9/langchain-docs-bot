from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load vector store
embedding = HuggingFaceEmbeddings()
db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)

# Load local LLM
llm = pipeline("text-generation", model="microsoft/phi-2", device="cpu")

# Session memory
import os

session_history = []

while True:
    print("\n====================")
    query = input("Ask your question (or type 'exit' to quit): ")
    print("====================")

    if query.lower() == "exit":
        print("Session ended. Goodbye!")
        break

    # Retrieve top chunk
    docs = db.similarity_search(query, k=3)
    context = docs[0].page_content if docs else ""

    # Build prompt with limited memory (last 3 turns)
    memory_block = "\n".join([f"Q: {q}\nA: {a}" for q, a in session_history[-3:]])
    prompt = (
        "You are a helpful assistant. Use the previous Q&A for context if relevant.\n\n"
        + memory_block
        + "\n\nCurrent excerpt:\n"
        + context
        + "\n\nQuestion: " + query + "\nAnswer: Please explain clearly in 3–5 sentences."
    )

    # Generate response
    response = llm(prompt, max_new_tokens=200)[0]['generated_text']
    # Trim unwanted leakage
    response = response.split("Question:")[0]
    response = response.replace(
        "You are a helpful assistant. Use the previous Q&A for context if relevant.", ""
    ).strip()

    # Fallback if empty
    if not response:
        response = "No answer generated. Try rephrasing your question."

    print("\nLLM Response:\n", response)

    # Save to session memory
    session_history.append((query, response))

    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)
    with open("logs/session_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n---\nQ: {query}\nA: {response}\n")

