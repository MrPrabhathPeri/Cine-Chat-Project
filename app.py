import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from groq import Groq
import requests

# --------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------
st.set_page_config(page_title="Cine-Chat", page_icon="🎬", layout="wide")

st.title("🎬 Cine-Chat: The AI Movie Expert")
st.caption("Powered by Llama 3.3 & RAG")

# --------------------------------------------------------------
# HELPER FUNCTIONS (OMDb Version)
# --------------------------------------------------------------
def fetch_poster(movie_title):
    """
    Searches OMDb for the movie title and returns the poster URL.
    """
    try:
        # We use st.secrets to keep the key safe
        api_key = st.secrets["OMDB_API_KEY"]
        
        # OMDb API Request (Using 't=' for Title Search)
        url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        # Check if a poster exists in the response
        if 'Poster' in data and data['Poster'] != 'N/A':
            return data['Poster']
        else:
            return None
    except:
        return None

# --------------------------------------------------------------
# SETUP (CACHED)
# --------------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except:
        st.error("GROQ_API_KEY not found in secrets!")
        st.stop()

    client = Groq(api_key=GROQ_API_KEY)

    db_path = "movie_db"
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="movies", embedding_function=sentence_transformer_ef)
    
    # BUILD DB IF EMPTY
    if collection.count() == 0:
        st.info("Building database... (~1 min)")
        if not os.path.exists('tmdb_5000_movies.csv'):
            st.error("CSV file not found. Please upload it!")
            st.stop()
            
        df = pd.read_csv('tmdb_5000_movies.csv')
        import json
        def extract_names(text):
            try: return " ".join([item['name'] for item in json.loads(text)])
            except: return ""
        
        df['combined_text'] = ("Genre: " + df['genres'].apply(extract_names) + 
                               " Keywords: " + df['keywords'].apply(extract_names) + 
                               " Plot: " + df['overview'].fillna(""))
        
        ids = [str(i) for i in df['id'].tolist()]
        documents = df['combined_text'].tolist()
        metadatas = df[['title', 'id']].to_dict(orient='records')
        
        batch_size = 200
        for i in range(0, len(df), batch_size):
            end = min(i + batch_size, len(df))
            collection.add(ids=ids[i:end], documents=documents[i:end], metadatas=metadatas[i:end])
        st.success("Database built!")
    
    return client, collection

client, collection = load_resources()

# --------------------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am a movie expert. Ask me anything!"}]

# Display History with Posters
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        col1, col2 = st.columns([1, 4]) # 1 part image, 4 parts text
        with col1:
            # Check if this message has a poster stored
            if "poster" in msg and msg["poster"]:
                st.image(msg["poster"], width=130)
        with col2:
            st.write(msg["content"])
    else:
        st.chat_message("user").write(msg["content"])

# Handle Input
if prompt := st.chat_input("Ask for a movie recommendation..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # RAG Retrieval
    results = collection.query(query_texts=[prompt], n_results=1) 
    top_movie = results['metadatas'][0][0]['title']
    top_plot = results['documents'][0][0]

    # Generate Answer
    system_prompt = f"""
    You are a movie expert. The user wants a recommendation.
    The best match is: {top_movie}.
    Plot: {top_plot}.
    
    Recommend this movie and explain why it fits the user's request.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    
    response = chat_completion.choices[0].message.content
    
    # Fetch Poster using OMDb
    poster_url = fetch_poster(top_movie)
    
    # Display Result
    col1, col2 = st.columns([1, 4])
    with col1:
        if poster_url:
            st.image(poster_url, width=130)
    with col2:
        st.write(response)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": response, "poster": poster_url})

