"""
Loads news article URLs from the sidebar, processes them to extract text, 
embeds the text with OpenAI embeddings, and saves the embeddings to a FAISS index.

The index can then be used to run QA via RetrievalQAWithSourcesChain to find answers
in the loaded articles.
"""

# Import necessary libraries 
import os
import streamlit as st
import pickle
import time
from langchain import OpenAI # Import OpenAI API wrapper
from langchain.chains import RetrievalQAWithSourcesChain # Import QA chain 
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text
from langchain.document_loaders import UnstructuredURLLoader # For loading URLs
from langchain.embeddings import OpenAIEmbeddings # For generating embeddings
from langchain.vectorstores import FAISS # For storing embeddings

# from dotenv import load_dotenv  
# load_dotenv()  # Load environment variables from .env file

# # Print OpenAI API key loaded from .env
# print(os.environ.get("OPENAI_API_KEY"))  

# Create Streamlit app and sidebar
st.title("Equity Research and Analysis Tool 📈")
st.sidebar.title("News Article URLs")

# Initialize empty list to store URLs  
urls = []

# Allow user to input 3 URLs in sidebar
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Print URLs  
print(urls)

# Button to trigger processing of URLs
process_url_clicked = st.sidebar.button("Process URLs")  

# File path to save FAISS index
file_path = "faiss_store_openai.pkl"  

# Placeholder to display status  
main_placeholder = st.empty()

# Initialize OpenAI API wrapper 
llm = OpenAI(openai_api_key="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", temperature=0.9, max_tokens=500)

# If process button clicked
if process_url_clicked:

    # Load URLs using loader
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split loaded text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Check if embeddings is empty
    if not docs:
        main_placeholder.text("Error: No documents found.")
    else:
        # Generate OpenAI embeddings 
        embeddings = OpenAIEmbeddings()
        
        # Check if embeddings generated properly
        if not embeddings or not embeddings[0]:
            main_placeholder.text("Error: 'embeddings' list is empty or the first element is empty.")
        else:
            # Create FAISS index to store embeddings
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)

            # Save FAISS index to pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

# Allow user to input question
query = main_placeholder.text_input("Question: ")

# If question entered
if query:
    # Check if FAISS index file exists
    if os.path.exists(file_path):
        # Load FAISS index
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            # Create QA chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            
            # Get answer from QA chain
            result = chain({"question": query}, return_only_outputs=True)
            
            # result will be a dictionary --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                
                # Split sources by newline
                sources_list = sources.split("\n")  
                for source in sources_list:
                    st.write(source)
