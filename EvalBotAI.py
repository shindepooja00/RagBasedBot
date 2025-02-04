import streamlit as st
import pandas as pd
import os
from pinecone import Pinecone, PineconeConfigurationError
from sentence_transformers import SentenceTransformer
import openai

# Load API Keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = "us-east-1"  # Change if necessary
INDEX_NAME = "rag-chatbot"

# Check if API keys are loaded correctly
if not OPENAI_API_KEY:
    st.error("OpenAI API key is missing! Set OPENAI_API_KEY as an environment variable.")
    st.stop()

openai.api_key = OPENAI_API_KEY

if not PINECONE_API_KEY:
    st.error("Pinecone API key is missing! Set PINECONE_API_KEY as an environment variable.")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(page_title="EvalutionBot - AI-powered Evolution Assistant", layout="wide")
st.title("🤖 EvalutionBot")
st.markdown("Hello! I'm **EvalutionBot**, your smart assistant for evalution-related queries. How can I help you today?")

# Sidebar Settings
st.sidebar.header("Advanced Settings")
top_k = st.sidebar.slider("Number of retrieval results (top_k):", min_value=1, max_value=10, value=3, step=1)
st.sidebar.markdown("[Explore iEval Discovery](https://webapps.ilo.org/ievaldiscovery/#b5b9for)")
if st.sidebar.button("Clear Conversation"):
    st.session_state.chat_history = []

# Load Data
@st.cache_data
def load_data():
    # Absolute path to the dataset
    file_path = r"C:\Users\Pooja\OneDrive\Documents\ilo-Chatbot\EvalDataset.xlsx"
    df = pd.read_excel(file_path)
    df = df.drop_duplicates().dropna()
    df.set_index(df.index.astype(str), inplace=True)
    return df

df = load_data()

# Load Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except PineconeConfigurationError as e:
    st.error(f"Pinecone configuration error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error initializing Pinecone: {e}")
    st.stop()

# Query Pinecone
def query_index(query_text, top_k):
    query_embedding = embed_model.encode(query_text, convert_to_numpy=True).tolist()
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return response

# Generate AI Response using OpenAI GPT (Updated for GPT-3.5)
def generate_response(query_text, retrieval_response):
    retrieved_info = []
    for match in retrieval_response.get("matches", []):
        match_id = match["id"]
        if match_id in df.index:
            retrieved_info.append(df.loc[match_id, "Evaluation title"])
    
    retrieved_text = "\n".join(retrieved_info) if retrieved_info else "No relevant evolution data found."
    
    prompt = f"""
    User Query: {query_text}
    Relevant Evolution Data:
    {retrieved_text}
    Based on the above, provide a detailed, factual response:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Updated to GPT-3.5
        messages=[{"role": "system", "content": "You are an expert AI assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=300
    )
    
    return response["choices"][0]["message"]["content"].strip()

# Build Interactive Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role, content = message["role"], message["content"]
    st.markdown(f"**{'You' if role == 'user' else 'EvaluationBot'}:** {content}")

user_input = st.text_input("Enter your question:")
if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Processing..."):
            retrieval_results = query_index(user_input, top_k)
            ai_response = generate_response(user_input, retrieval_results)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        st.rerun()
    else:
        st.warning("Please enter a query.")
