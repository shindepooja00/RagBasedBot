# EvaluationBot 🤖  
**A Retrieval-Augmented Generation (RAG)-based chatbot using OpenAI, Pinecone, and Streamlit for evaluation-related queries.**  

---

## 📌 Project Objectives  
EvaluationBot is designed to:  
- **Assist users** with evaluation-related queries using AI.  
- **Retrieve relevant data** from a structured dataset using vector search.  
- **Generate accurate, contextual responses** leveraging OpenAI's GPT model.  
- **Enhance user interaction** via a simple and intuitive Streamlit UI.  
- **Guide users to iEval Discovery website** for further exploration.  

---

## 🏗️ **Architecture & Models Used**  

### **1️⃣ Retrieval-Augmented Generation (RAG) Pipeline**  
- **Data Retrieval:** Pinecone (Vector Search)  
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)  
- **Response Generation:** OpenAI GPT-3.5 Turbo  

### **2️⃣ Workflow**  
1. **User Input:** A query is entered in the chatbot UI.  
2. **Vector Search:** The input is converted into an embedding and matched with stored vectors in Pinecone.  
3. **Relevant Data Retrieval:** Matching evaluation data is extracted from the dataset.  
4. **AI Response Generation:** OpenAI GPT-3.5 generates a response using retrieved information.  
5. **User Guidance:** The chatbot provides direct answers and guides users to external resources if necessary.  

---

## 🛠 **Technologies Used**  
| Technology       | Purpose |  
|-----------------|---------|  
| **Python** | Backend logic & AI processing |  
| **Streamlit** | Interactive chatbot UI |  
| **OpenAI GPT-3.5 Turbo** | AI-based response generation |  
| **Pinecone** | Vector database for efficient retrieval |  
| **SentenceTransformers** | Text embeddings for vector search |  
| **Pandas** | Data handling and preprocessing |  
| **Excel Dataset** | Structured knowledge base |  

---

## 🧠 **Required Skill Sets**  
To understand, modify, or extend this project, you should be familiar with:  
✅ **Python** – Writing scripts and backend logic.  
✅ **NLP (Natural Language Processing)** – Understanding embeddings, vector databases, and text generation.  
✅ **Machine Learning** – Working with transformer models and vector search techniques.  
✅ **OpenAI API** – Generating responses using GPT-3.5.  
✅ **Pinecone** – Managing vector embeddings and performing similarity search.  
✅ **Streamlit** – Building interactive web applications.  
✅ **Pandas** – Handling and processing tabular data.  

---

## 🚀 **Getting Started**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/EvaluationBot.git
cd EvaluationBot
