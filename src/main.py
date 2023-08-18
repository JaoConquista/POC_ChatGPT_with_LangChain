import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

#chat with memory
from IPython.display import display
import ipywidgets as widgets


load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Read PDF and split into pages
loader = PyPDFLoader("../data/your_file.pdf")
pages = loader.load_and_split()

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Check similarity search is working
query = "Check if the search is working"
docs = db.similarity_search(query)
docs[0]

# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.7, streaming=True), db.as_retriever())

chat_history = []

print("Welcome to the Transformers chatbot!")

while True:
    query = input("Please enter your question about the document: ")
    
    if query.lower() == 'exit':
        print("Thank you for using the chatbot!")
        break
    
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    print(f'\033[33mUser: {query}\033[0m') 
    print(f'\033[34mChatbot: {result["answer"]}\033[0m')
    #If you want to see the full context, uncomment the line below
    #print(f'Font: {docs}')