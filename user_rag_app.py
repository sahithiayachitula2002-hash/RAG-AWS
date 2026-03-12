import os
import json
import uuid
import faiss
import boto3
import pickle
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "us-east-1"

LLM_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("chat_logs", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)
os.makedirs("uploads", exist_ok=True)


# ============================================================
# USER MANAGEMENT FUNCTIONS
# ============================================================

def load_users():
    if os.path.exists("users.json"):
        with open("users.json") as f:
            return json.load(f)
    return {}


def save_users(data):
    with open("users.json", "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def get_embedding(text):

    body = json.dumps({"inputText": text})

    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=body
    )

    response_body = json.loads(response["body"].read())

    return np.array(response_body["embedding"], dtype=np.float32)


# ============================================================
# LLM FUNCTION
# ============================================================

def call_llm(prompt):

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    })

    response = bedrock.invoke_model(
        modelId=LLM_MODEL,
        body=body
    )

    response_body = json.loads(response["body"].read())

    return response_body["content"][0]["text"]


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text, size=1000, overlap=200):

    chunks = []
    start = 0

    while start < len(text):

        end = start + size
        chunks.append(text[start:end])

        start = end - overlap

    return chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(text):

    chunks = chunk_text(text)

    embeddings = []

    for chunk in chunks:
        embeddings.append(get_embedding(chunk))

    embeddings = np.vstack(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "faiss_index/index.faiss")

    with open("faiss_index/texts.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_index():

    index_path = "faiss_index/index.faiss"
    text_path = "faiss_index/texts.pkl"

    if os.path.exists(index_path) and os.path.exists(text_path):

        index = faiss.read_index(index_path)

        with open(text_path, "rb") as f:
            docs = pickle.load(f)

        return index, docs

    return None, []


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve(query, index, docs, k=3):

    if index is None:
        return []

    vec = get_embedding(query).reshape(1, -1)

    distances, indices = index.search(vec, k)

    results = []

    for i in indices[0]:

        if i < len(docs):
            results.append(docs[i])

    return results


# ============================================================
# SAVE CHAT
# ============================================================

def save_user_chat(chat_id, username, conversation):

    if not chat_id:
        return

    path = f"chat_logs/{chat_id}.json"

    data = {
        "username": username,
        "conversation": conversation
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# LOAD CHAT
# ============================================================

def load_chat(chat_id):

    path = f"chat_logs/{chat_id}.json"

    if os.path.exists(path):

        with open(path) as f:
            data = json.load(f)

        return data.get("conversation", [])

    return []


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, conversation, docs):

    context = "\n".join(docs)

    history = ""

    for turn in conversation[-20:]:

        history += f"User: {turn['user']}\nBot: {turn['bot']}\n"

    prompt = f"""
You are a helpful assistant.

Use the conversation history to answer follow-up questions.

Use ONLY the provided document context.

Conversation:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

    return call_llm(prompt)


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot (AWS Bedrock)")


# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "index" not in st.session_state:
    st.session_state.index = None

if "docs" not in st.session_state:
    st.session_state.docs = []

if "chats" not in st.session_state:
    st.session_state.chats = []


users = load_users()


# ============================================================
# LOAD VECTOR INDEX INTO SESSION
# ============================================================

if st.session_state.index is None:

    index, docs = load_index()

    if index is not None:

        st.session_state.index = index
        st.session_state.docs = docs


# ============================================================
# LOGIN UI
# ============================================================

if not st.session_state.authenticated:

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in users:

            if users[username]["password"] == password:

                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.chats = users[username].get("chats", [])

                st.success("Login successful")
                st.rerun()

            else:
                st.error("Incorrect password")

        else:

            users[username] = {
                "password": password,
                "chats": []
            }

            save_users(users)

            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.chats = []

            st.success("New user created")
            st.rerun()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.write(f"👤 {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("## Chats")


# ============================================================
# NEW CHAT
# ============================================================

if st.sidebar.button("➕ New Chat"):

    new_chat_id = str(uuid.uuid4())

    chat_info = {
        "chat_id": new_chat_id,
        "title": f"Chat {len(st.session_state.chats)+1}"
    }

    st.session_state.chats.append(chat_info)

    users[st.session_state.username]["chats"] = st.session_state.chats
    save_users(users)

    st.session_state.chat_id = new_chat_id
    st.session_state.conversation = []

    st.rerun()


# ============================================================
# CHAT LIST
# ============================================================

for chat in st.session_state.chats:

    if st.sidebar.button(chat["title"], key=chat["chat_id"]):

        st.session_state.chat_id = chat["chat_id"]

        st.session_state.conversation = load_chat(chat["chat_id"])

        st.rerun()


# ============================================================
# PDF UPLOAD UI
# ============================================================

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        t = page.extract_text()

        if t:
            text += t

    with st.spinner("Creating embeddings..."):

        index, docs = build_index(text)

        st.session_state.index = index
        st.session_state.docs = docs

    st.success("PDF indexed")


# ============================================================
# CHAT UI
# ============================================================

if st.session_state.chat_id:
    
    with st.form("chat_form", clear_on_submit=True):
        
        user_input = st.text_input("Ask question")
        submit = st.form_submit_button("Submit")


    if submit and user_input:

        docs = retrieve(
            user_input,
            st.session_state.index,
            st.session_state.docs
        )

        answer = generate_answer(
            user_input,
            st.session_state.conversation,
            docs
        )

        chat = {
            "time": str(datetime.now()),
            "user": user_input,
            "bot": answer
        }

        st.session_state.conversation.append(chat)

        save_user_chat(
            st.session_state.chat_id,
            st.session_state.username,
            st.session_state.conversation
        )

        st.write("### Answer")
        st.write(answer)


# ============================================================
# CHAT HISTORY UI
# ============================================================

if st.session_state.conversation:

    st.markdown("### Chat History")

    for chat in reversed(st.session_state.conversation):

        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
        