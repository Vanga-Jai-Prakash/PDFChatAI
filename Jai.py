import streamlit as st #for UI use
from PyPDF2 import PdfReader #for reading pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter #for splitting


from sentence_transformers import SentenceTransformer #for transformer
import faiss #for storing vectors
import numpy as np

from groq import Groq

groq_client = Groq(api_key="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")


embedder = SentenceTransformer("all-MiniLM-L6-v2") #free embedding model key

#Upload pdf
st.header("CHATBOT")
with st.sidebar:
    st.title("CHATBOT")
    file = st.file_uploader("Upload your file & ask question", type= "pdf")

#EXTRACTING TEXT
if file is not None:
    Pdf_Reader = PdfReader(file)
    text = ""
    for page in Pdf_Reader.pages:
        text += page.extract_text()


#Breaking the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
    separators= "\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function= len
    )

    chunks = text_splitter.split_text(text)
    #st.write(chunks)

# Generating Embeddings
    embeddings = embedder.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.success("PDF processed successfully! Ask your question below.")

    # -----------------------------
    #  USER QUESTION
    # -----------------------------
    query = st.text_input("Ask a question about the PDF:")

    if query:
        # Convert query → embedding
        query_embedding = embedder.encode([query])

        # FAISS search
        k = 3
        distances, indices = index.search(np.array(query_embedding), k)

        # Retrieve relevant chunks
        retrieved_text = "\n".join([chunks[i] for i in indices[0]])

        # -----------------------------
        prompt = f"""
        You are an AI assistant. Answer the user's question using the PDF text below.
        PDF text:
        {retrieved_text}

        Question: {query}

        Answer:
        """

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        st.subheader("Answer:")
        st.write(answer)