# 🎬 Cine-Chat: AI-Powered Movie Assistant

**Cine-Chat** is a Retrieval-Augmented Generation (RAG) application that provides context-aware movie recommendations. Unlike traditional keyword search, it uses **Semantic Vector Search** to understand the *meaning* of a user's request (e.g., "sad movies about space") and generates explained recommendations using **Llama 3**.

🔗 **Live Demo:** [Click Here to Chat!](https://share.streamlit.io/YOUR_GITHUB_USERNAME/cine-chat/main)  
*(Replace with your actual Streamlit Cloud link)*

---

## 🚀 Key Features
* **🧠 RAG Architecture:** Combines vector search (ChromaDB) with a Large Language Model (Llama 3.3 via Groq) to ground answers in factual data.
* **🔍 Semantic Search:** Uses `all-MiniLM-L6-v2` embeddings to find movies based on plot themes, not just keywords.
* **⚡ Ultra-Fast Inference:** Optimized using Groq's LPU (Language Processing Unit) for near-instant AI responses.
* **🎨 Dynamic Visuals:** Fetches real-time movie posters using the OMDb API.
* **🛠 Self-Healing Database:** Automatically builds/rebuilds the vector index on first launch, removing the need to upload heavy database files.

---

## 🛠️ Tech Stack
* **LLM:** Llama 3.3 (70B parameters)
* **Vector Database:** ChromaDB
* **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Frontend:** Streamlit
* **APIs:** Groq (Inference), OMDb (Posters)
* **Language:** Python 3.10+

---

## ⚙️ How It Works
1.  **Ingestion:** The app loads the `tmdb_5000_movies` dataset and creates a "soup" of metadata (Plot + Genre + Keywords).
2.  **Embedding:** This text is converted into 384-dimensional vectors and stored in **ChromaDB**.
3.  **Retrieval:** When a user asks a question, the system finds the 3 nearest vectors (most similar movies).
4.  **Augmentation:** These movie plots are injected into a prompt for **Llama 3**.
5.  **Generation:** Llama 3 explains *why* these movies fit the user's request, and the app fetches their posters.

---

## 📦 Installation (Local)
To run this project on your own machine:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/cine-chat.git](https://github.com/YOUR_USERNAME/cine-chat.git)
    cd cine-chat
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys**
    Create a `.streamlit/secrets.toml` file and add your keys:
    ```toml
    GROQ_API_KEY = "your_groq_key"
    OMDB_API_KEY = "your_omdb_key"
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 🔮 Future Improvements
* Add **Chat History Persistence** (save chats to a database).
* Implement **Hybrid Search** (Keyword + Vector) for better accuracy on specific actor names.
* Add **User Ratings** to filter movies by quality.
