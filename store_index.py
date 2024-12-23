from src.helper import load_pdf, text_splitter, download_hugging_face_embedings
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

extracted_data = load_pdf("data/")
text_chunks = text_splitter(extracted_data)
embeddings = download_hugging_face_embedings()

# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Extract text content from your text chunks
texts = [t.page_content for t in text_chunks]

# Define your Pinecone index name
index_name = "medical-chat-bot"

# Create the Pinecone vector store
docsearch = LangchainPinecone.from_texts(
    texts=texts,
    embedding=embeddings,
    index_name=index_name
)


