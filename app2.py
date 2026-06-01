import os

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

# -----------------------------------
# STEP 1: Get URL from user
# -----------------------------------

url = input("Enter Website URL: ")

# -----------------------------------
# STEP 2: Create unique folder name
# -----------------------------------

safe_folder_name = (
   url.replace("https://", "")
       .replace("http://", "")
       .replace("/", "_")
       .replace(".", "_")
)

print(safe_folder_name)

faiss_path = f"vector_db/{safe_folder_name}"

# -----------------------------------
# STEP 3: Create Embedding Model
# -----------------------------------

embeddings = OpenAIEmbeddings(
   model="text-embedding-3-small"
)

# -----------------------------------
# STEP 4: Load Existing FAISS
# -----------------------------------

if os.path.exists(faiss_path):

   print("\nLoading existing vector database...\n")

   vector_store = FAISS.load_local(
      faiss_path,
      embeddings,
      allow_dangerous_deserialization=True
   )

# -----------------------------------
# STEP 5: Create New FAISS
# -----------------------------------

else:

   print("\nWebsite not processed before.")
   print("Processing website...\n")

   # load website
   loader = WebBaseLoader(url)

   documents = loader.load()

   print(f"Documents Loaded: {len(documents)}")

   # chunking
   splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
   )

   chunks = splitter.split_documents(
      documents
   )
   
   print(f"Chunks Created: {len(chunks)}")

   # create vector store
   vector_store = FAISS.from_documents(
      chunks,
      embeddings
   )

   # save locally
   os.makedirs("vector_db", exist_ok=True)

   vector_store.save_local(
      faiss_path
   )

   print("\nVector database saved successfully.\n")

# -----------------------------------
# STEP 6: Create LLM
# -----------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini"
)

# -----------------------------------
# STEP 7: Chat Loop
# -----------------------------------

while True:

   user_query = input(
        "\nAsk a question (type 'exit' to quit): "
   )

   if user_query.lower() == "exit":
      print("\nGoodbye!")
      break

   # Retrieve relevant chunks
   results = vector_store.similarity_search(
      user_query,
      k=3
   )

   # Build context
   context = "\n\n".join(
      doc.page_content
      for doc in results
   )

   # prompt
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
   
   # generate response
   response = llm.invoke(prompt)

   print("\n" + "=" * 60)
   print("ANSWER")
   print("=" * 60)

   print(response.content)

   # Display Sources
   print("\nSOURCES")
   print("=" * 60)

   sources = set()

   for doc in results:
      source = doc.metadata.get(
         "source",
         "Unknown Source"
      )

      sources.add(source)

   for source in sources:
      print(source)
   
