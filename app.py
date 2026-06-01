from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

url = "https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/"

loader = WebBaseLoader(url)
documents = loader.load()

# print("\n\n processed")
# print(len(documents))
# print(type(documents))
# print(type(documents[0]))
# print(documents[0].metadata)
# print(len(documents[0].page_content))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)

# print(len(chunks))
# print(chunks[0].metadata)
# print(chunks[2].metadata)
# print(chunks[0].page_content)
# print(chunks[1].page_content)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(
   chunks,
   embeddings
)
# print(type(vector_store))
# print(vector_store.index.ntotal)   --> this gives number of vectors stored

print("vector db created")

user_query = input("Ask your question: ")

results = vector_store.similarity_search(
   user_query,
   k=3
)
# print(results[0].metadata)

# for i, doc in enumerate(results):
#     print(f"\nChunk {i+1}")
#     print("=" * 50)
#     print(doc.page_content[:300])
#     print("\nMetadata:")
#     print(doc.metadata)

context = "\n\n".join(
    doc.page_content
    for doc in results
)

prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided context."

Do not make up information.
Do not use your own knowledge.

Context:
{context}

Question:
{user_query}
"""

llm = ChatOpenAI(model="gpt-4.1-mini")

response = llm.invoke(prompt)

# print(response.content)



