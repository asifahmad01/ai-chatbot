import os

from flask import Flask, request, jsonify, render_template
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from flask_cors import CORS
from dotenv import load_dotenv
from pathlib import Path



load_dotenv(Path("./.env"))
app = Flask(__name__)

CORS(app)
# Set API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the model
#chatgpt = ChatOpenAI(model_name='gpt-3')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json['userText']


    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    vectordb = Chroma(persist_directory = 'vectorStore' ,collection_name="my_collection",embedding_function = OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff",retriever = retriever)
    response = qa.run(query)

    return jsonify({'botResponse': response})

if __name__ == "__main__":
    app.run(port=5000)
