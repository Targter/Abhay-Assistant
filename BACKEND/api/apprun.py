from flask import Flask, request, jsonify
from waitress import serve
import os
import logging
from functools import lru_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pyht import Client
from pyht.client import TTSOptions
from langchain.output_parsers import StructuredOutputParser

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing environment variables. Check your .env file.")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize TTS client
client = Client(
    user_id=os.getenv("PLAY_HT_USER_ID"),
    api_key=os.getenv("PLAY_HT_API_KEY"),
)
options = TTSOptions(voice="s3://voice-cloning-zero-shot/e7e9514f-5ffc-4699-a958-3627151559d9/nolansaad2/manifest.json")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Moves one level up
FILE_PATH = os.path.join(BASE_DIR, "data/file.txt")
def get_text_chunks():
    try:
        with open(FILE_PATH, "r", encoding="utf-8") as file:
            text = file.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return splitter.split_text(text)
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return []

def get_vector_store(text_chunks):
    try:
        if not os.path.exists("faiss_index"):
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            logging.info("Vector store created and saved.")
        else:
            logging.info("Vector store already exists. Skipping creation.")
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")

@lru_cache(maxsize=100)
def get_conversational_chain():
    prompt_template = """
    You are an assistant providing information on behalf of Abhay Bansal. Your responses should be clear, concise, and professional. Follow these guidelines:

1. **General Knowledge Questions**:
   - Provide a **detailed yet concise answer** based on general knowledge.
   - Ensure the response is informative and accurate.

2. **Questions Related to Abhay Bansal**:
   - If the question is about Abhay Bansal and the information is available, provide a **clear and concise answer**.
   - If the question is about Abhay Bansal but the information is not available, respond with:
     "I am unsure about this. For more information, you can connect with Abhay Bansal on LinkedIn: https://www.linkedin.com/in/bansalabhay/"

3. **Response Format**:
   - Keep the response **within 200 tokens**.
   - Maintain a **professional and clear** tone.

Context:\n{context}\n
Question:\n{question}\n

Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3 ,max_outputs_tokens = 200)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return create_stuff_documents_chain(llm=model, prompt=prompt)

BASIC_QUESTIONS = {
    "hi": "Hi! I'm Abhay Bansal. How can I help you today?",
    "hello": "Hello! I'm Abhay Bansal. What can I do for you?",
    "what is your name?": "My name is Abhay Bansal.",
    "who are you?": "I am Abhay Bansal. Nice to meet you!",
    "how are you?": "I'm doing great, thank you! How about you?",
    "abhay": "Yes, that's me! How can I help you today?",
}

def is_basic_question(question):
    question_lower = question.lower().strip()
    return question_lower in BASIC_QUESTIONS

def user_input(user_question):
    if is_basic_question(user_question):
        response = BASIC_QUESTIONS.get(user_question.lower().strip(), "I'm not sure how to answer that.")
        return response
    
    if os.path.exists("faiss_index"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    else:
        logging.error("FAISS index not found. Please create the vector store first.")
        return "FAISS index not found. Please create the vector store first."

    if not docs:
        logging.info("No relevant context found. Using general knowledge...")
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        response = model.invoke(user_question)
        return response
    
    chain = get_conversational_chain()
    response = chain.invoke({"context": docs, "question": user_question})
    return response

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_question = data.get('question')
        if not user_question:
            return jsonify({"error": "No question provided"}), 400
        
        response = user_input(user_question)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

def main():
    if not os.path.exists("faiss_index"):
        text_chunks = get_text_chunks()
        get_vector_store(text_chunks)

    # Serve the app using Waitress
    serve(app, host='0.0.0.0', port=8080)

if __name__ == "__main__":
    main()