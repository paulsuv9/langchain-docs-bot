from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import os


# Load and split
#loader = TextLoader("datasets/agentic_ai.txt")
#docs = loader.load()


# Load all .txt files manually using TextLoader
docs = []
for filename in os.listdir("docs/langchain"):
    if filename.endswith(".txt"):
        path = os.path.join("docs/langchain", filename)
        loader = TextLoader(path)
        docs.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Embed and store
embedding = HuggingFaceEmbeddings()
db = FAISS.from_documents(chunks, embedding)
db.save_local("vector_store")
