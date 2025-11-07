from transformers import pipeline

llm = pipeline("text-generation", model="microsoft/phi-2")
response = llm("What is Agentic AI?", max_new_tokens=100)
print(response[0]["generated_text"])


