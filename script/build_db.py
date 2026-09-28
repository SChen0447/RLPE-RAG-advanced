from app.rag.vectorstore import VectorStore
from app.rag.loaders import load_documents, split_documents, get_file_paths

vectordb = VectorStore()

file_paths = get_file_paths()
documents = load_documents(file_paths)
chunks = split_documents(documents)

vectordb.build(chunks)
retriever = vectordb.get_vectorstore().as_retriever()

question = "什么是提示词工程"

result = retriever.invoke(question)

print(result)

