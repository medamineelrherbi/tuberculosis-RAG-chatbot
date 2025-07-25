import os
import pdfplumber
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


# === CONFIG ===
PDF_FOLDER = "C:/Users/UTILISATEUR/Desktop/RAG/tuberculous"
FAISS_INDEX_PATH = "C:/Users/UTILISATEUR/Desktop/RAG/vectorstore/"
DOC_INDEX_FILE = "C:/Users/UTILISATEUR/Desktop/RAG/pdf_index.json"

# === Custom Embedding Class ===
class LocalEmbeddingModel(Embeddings):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return [np.array(e).tolist() for e in self.model.encode(texts, show_progress_bar=True)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# === Load PDF documents ===
def load_pdf_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            with pdfplumber.open(path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                docs.append((filename, Document(page_content=text)))
    return docs

# === Check indexed documents ===
def load_indexed_files():
    if os.path.exists(DOC_INDEX_FILE):
        with open(DOC_INDEX_FILE, "r") as f:
            return json.load(f)
    return []

def save_indexed_files(indexed_files):
    with open(DOC_INDEX_FILE, "w") as f:
        json.dump(indexed_files, f)

# === Prepare Vectorstore ===
def prepare_vectorstore():
    embedding_model = LocalEmbeddingModel()
    indexed_files = load_indexed_files()
    docs_to_index = []

    all_docs = load_pdf_documents(PDF_FOLDER)
    for filename, doc in all_docs:
        if filename not in indexed_files:
            docs_to_index.append((filename, doc))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    # === Si FAISS existe déjà, on le charge
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

    # === Sinon, on ne crée l'index QUE s'il y a des documents à indexer
    elif docs_to_index:
        all_chunks = []
        for _, doc in docs_to_index:
            all_chunks.extend(splitter.split_documents([doc]))
        vectorstore = FAISS.from_documents(all_chunks, embedding_model)
        for filename, _ in docs_to_index:
            indexed_files.append(filename)
        vectorstore.save_local(FAISS_INDEX_PATH)
        save_indexed_files(indexed_files)
        print(f"✅ Indexed {len(docs_to_index)} new PDFs.")

    else:
        raise ValueError("❌ No existing FAISS index and no new documents to index.")

    return vectorstore, embedding_model


# === Groq LLM Call ===
def call_groq_llm(prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f" Error calling LLM API: {str(e)}"

# === Prompt template ===
def build_medical_prompt(context, question):
    return f"""
You are a helpful and knowledgeable medical assistant. 
Answer the user's question as clearly and informatively as possible, based only on the provided context.

If the answer isn't clearly in the context, say "I don't know based on the documents."

Use simple, understandable language suitable for a patient or a student.

Context:
{context}

Question: {question}
Answer:
""".strip()

# === RAG Chat Function ===
def answer_question(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    if len(context) > 6000:
        context = context[:6000]
    prompt = build_medical_prompt(context, question)
    return call_groq_llm(prompt)


#application flask

app = Flask(__name__)
CORS(app)  # Autorise les requêtes JS depuis le HTML

vectorstore, _ = prepare_vectorstore()

@app.route("/")
def index():
    return render_template("index.html")  # Optionnel si tu veux servir l'UI via Flask

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("message")
    if not user_input:
        return jsonify({"answer": "Please enter a valid question."}), 400

    answer = answer_question(vectorstore, user_input)
    return jsonify({"answer": answer})
# === MAIN LOOP ===
if __name__ == "__main__":
    app.run(debug=True)

        
