from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
import os
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings
)
import sqlite3
from flask import Flask, request, jsonify
from functools import wraps
from shutil import rmtree
from datetime import datetime
from dotenv import load_dotenv
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
import faiss
import openai
from google.generativeai import configure, GenerativeModel
import google.generativeai as genai
load_dotenv()
#embed_model = HuggingFaceEmbedding("localmodels/all-MiniLM-L6-v2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
embed_model = HuggingFaceEmbedding("localmodels/all-MiniLM-L6-v2")

Settings.embed_model=embed_model
sentence_splitter = split_by_sentence_tokenizer()
splitter = SemanticSplitterNodeParser(buffer_size=1,breakpoint_percentile_threshold=75,sentence_splitter=sentence_splitter,include_metadata=True,include_prev_next_rel=True,embed_model=embed_model)
docstore = SimpleDocumentStore()

DATA_DIR = "./data"
STORAGE_DIR = "./storage"
INDEX_DIR = "./index"
DOCS_DIR = "./docstore/docstore.json"
DOCS_MAIN_DIR = "./docstore"

os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(STORAGE_DIR,exist_ok=True)
os.makedirs(INDEX_DIR,exist_ok=True)
os.makedirs(DOCS_MAIN_DIR,exist_ok=True)

def require_admin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != ADMIN_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


app = Flask(__name__)


#documents = SimpleDirectoryReader("./data").load_data()
def init_db():
    conn = sqlite3.connect('usage_logs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_logs
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp TEXT,
         endpoint TEXT,
         query TEXT,
         response TEXT)
    ''')
    conn.commit()
    conn.close()

init_db()

def log_usage(endpoint, query, response):
    conn = sqlite3.connect('usage_logs.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_logs (timestamp, endpoint, query, response)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now().isoformat(), endpoint, query, str(response)))
    conn.commit()
    conn.close()

def retrieve_nodes_top_k(query,k=5):
    
    list_of_node_ids = []
    query_embedding = embed_model.get_text_embedding(query)
    vector_store = FaissVectorStore.from_persist_dir(INDEX_DIR)
    
    vector_query = VectorStoreQuery(query_embedding=query_embedding,similarity_top_k=k)
    retrieved_nodes = vector_store.query(vector_query)
    for similarity, node_id in zip(retrieved_nodes.similarities,retrieved_nodes.ids):
        print("Node ID:", node_id)
        print(type(node_id))
        list_of_node_ids.append((node_id,similarity))
    return list_of_node_ids

def retrieve_nodes_data(nodes_from_index):
    node_ids = []
    import pickle
    with open(os.path.join(INDEX_DIR, 'index_to_node_id_mapping.pkl'), 'rb') as f:
        index_to_node_id = pickle.load(f)
    for i in range(len(nodes_from_index)):
        node_ids.append(nodes_from_index[i][0])
    print(node_ids)
    mapped_node_ids = [index_to_node_id.get(idx, None) for idx in node_ids]
    print(mapped_node_ids)
    mapped_node_ids = [node_id for node_id in mapped_node_ids if node_id is not None]
    print(mapped_node_ids)
    docstore = SimpleDocumentStore.from_persist_path(DOCS_DIR)
    
    retrieved_nodes = docstore.get_nodes(mapped_node_ids)
    return retrieved_nodes

def create_faiss():
    print("Creating a new vector store index...")
    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
    except:
        print("No documents found in the directory. Please make sure it's not empty!")
        return None
    total_sentences = 0
    doc_ids = list(str(i) for i in range(len(documents)))
    print(len(doc_ids))
    persist_dir = INDEX_DIR
    for doc in documents:
        sentences = sentence_splitter(doc.text)
        total_sentences+=len(sentences)
    for doc, doc_id in zip(documents, doc_ids):
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["document_id"] = doc_id
    print("document 1 metadata:")
    print(documents[1].metadata)
    nodes = splitter.get_nodes_from_documents(documents)
    docstore.add_documents(nodes)
    docstore.persist(persist_path=DOCS_DIR)
    print("total sentences:"+str(total_sentences))
    print("total chunks:"+str(len(nodes)))
    print("sample node:")
    print(nodes[1].get_content())
    
    print("sample node metadata:")
    print(nodes[1].node_id)
    print(nodes[1].metadata)
    print(type(nodes[1].metadata))
    index_to_node_id = {}
    node_id_to_file = {file: [] for file in os.listdir(DATA_DIR)}
    try:
        for idx,node in enumerate(nodes):
            node_embedding = embed_model.get_text_embedding(node.get_content())
            node.embedding = node_embedding
            index_to_node_id[str(idx)] = node.node_id
            node_id_to_file[node.metadata.get('file_name')].extend([node.node_id])
        faiss_index = faiss.IndexFlatIP(384)
        print(nodes[12].embedding)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        vector_store.add(nodes)
        print(nodes[12].node_id)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        storage_context.persist(persist_dir=INDEX_DIR)
        print(index_to_node_id)
        print(node_id_to_file)
        import pickle
        with open(os.path.join(INDEX_DIR, 'index_to_node_id_mapping.pkl'), 'wb') as f:
            pickle.dump(index_to_node_id, f)
        with open(os.path.join(INDEX_DIR, 'node_id_to_file_name.pkl'), 'wb') as f:
            pickle.dump(node_id_to_file, f)
        print(f"FAISS index saved at: {INDEX_DIR}")
        print("index saved successfully!")
        return True
    except:
        return False

create_faiss()
    
def retrieve_context(question):
    list_of_nodes = retrieve_nodes_top_k(question)
    node_sources = []
    retrieved_nodes = retrieve_nodes_data(list_of_nodes)
    retrieved_text = []
    for node in retrieved_nodes:
        retrieved_text.append(node.get_content())
        node_sources.append(("Source file:"+str(node.metadata.get("file_name")),"Source document ID: "+str(node.metadata.get("document_id"))))
    return retrieved_text, node_sources

def generate_answer(text,query):
    configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Create the prompt
    prompt = f"""Based on the following context, answer the question.
    If the context doesn't contain relevant information, say "I don't have enough information to answer that question."
    
    Context:
    {text}
    
    Question: {query}
    
    Answer:""".format(text=text,query=query)

    try:
        # Load Gemini Pro model
        model = GenerativeModel('gemini-pro')
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@app.route('/admin/upload',methods=['POST'])
@require_admin
def upload_doc_from_user():
    try:
        if 'documents' not in request.files:
            return jsonify({"error": "No documents provided"}), 400
            
        files = request.files.getlist('documents')
        
        for file in files:
            file_path = os.path.join(DATA_DIR, file.filename)
            file.save(file_path)
        
        create_faiss()
        log_usage("UPLOAD","(UPLOADED A FILE)","ADDED THE FILE TO THE DOCSTORE")
        return jsonify({"message":"Documents uploaded successfully, and created the index for the new documents."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    ""

@app.route('/admin/log',methods=['GET'])
@require_admin
def admin_check_logs():
    try:
        conn = sqlite3.connect('usage_logs.db')
        c = conn.cursor()
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date and end_date:
            c.execute('''
                SELECT * FROM user_logs 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (start_date, end_date))
        else:
            c.execute('SELECT * FROM user_logs ORDER BY timestamp DESC')
            
        logs = c.fetchall()
        conn.close()
        
        return jsonify([{
            "id": log[0],
            "timestamp": log[1],
            "endpoint": log[2],
            "query": log[3],
            "response": log[4]
        } for log in logs])
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/query',methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        text, src = retrieve_context(query)
        answer = generate_answer(text,query)
        log_usage("QUERY",query,answer)
        return jsonify({"repsonse":answer,"sources":src})
    except Exception as e:
        return jsonify({"error":"Error in responding to this query:"+str(e)})

if __name__ == "__main__":
    app.run(port=5100,debug=True)