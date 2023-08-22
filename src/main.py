import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Read PDF and split into pages
counter = 0
chunks = []
for file in os.listdir("../data"):
    counter += 1
    print(f"\033[32mfile_{counter}: {file}\033[0m")
    loader = PyPDFLoader(f"../data/{file}")
    pages = loader.load_and_split()
    chunks = pages + chunks
print(chunks)

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5, streaming=True), db.as_retriever())

chat_history = []

print("Welcome to the Transformers chatbot!")

while True:
    query = input("Please enter your question about the document: ")
    docs = db.similarity_search(query)
    
    if query.lower() == 'exit':
        print("Thank you for using the chatbot!")
        break
    
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    print(f'\033[33mUser: {query}\033[0m') 
    print(f'\033[34mChatbot: {result["answer"]}\033[0m')
    #If you want to see the full context, uncomment the line below
    print(f'Font: {docs}')