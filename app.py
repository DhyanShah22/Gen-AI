import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Streamlit configuration
st.set_page_config(page_title="Chat with PDF", page_icon="📚")

# Set up the Gemini API key
GEMINI_API_KEY = "AIzaSyBFoHj9lZ4soXr0l-GDhg8TJO6PwRs0yoo"

# Cache document loading and vector store creation
@st.cache_resource(show_spinner=False)
def load_data():
    try:
        with st.spinner("Loading and indexing the document..."):
            # Check if data directory exists
            if not os.path.exists("./data"):
                st.error("❌ Data directory not found! Creating one...")
                os.makedirs("./data")
                st.info("📁 Created ./data directory. Please add your PDF file there.")
                st.stop()
            
            # List files in data directory
            files = os.listdir("./data")
            pdf_files = [f for f in files if f.endswith('.pdf')]
            
            if not pdf_files:
                st.error("❌ No PDF files found in the data directory!")
                st.info("ℹ️ Please add a PDF file to the ./data directory")
                st.stop()
            
            st.info(f"📄 Found file: {pdf_files[0]}")
            
            # Load PDF document
            pdf_path = os.path.join("./data", pdf_files[0])
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                st.error("❌ Could not load the PDF document!")
                st.stop()
            
            st.success(f"✅ Successfully loaded PDF with {len(documents)} pages")
            
            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY,
                credentials=None
            )
            
            # Create vector store
            vector_store = FAISS.from_documents(documents, embeddings)
            st.success("✅ Successfully created vector store!")
            
            return vector_store
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# Load the vector store
vector_store = load_data()

# Set up the conversational retrieval chain
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",  # Specify which key to store
    return_messages=True
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
    credentials=None,
    convert_system_message_to_human=True
)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    chain_type="stuff",
    verbose=True
)

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions about the PDF document..."}
    ]

# Streamlit user interface
st.title("Chat With PDF")
st.caption("Ask questions about your PDF document")

# Sidebar content
with st.sidebar:
    st.write("Debug Information")
    st.write("Messages in state:", len(st.session_state.messages))
    st.write("Memory exists:", memory is not None)
    st.write("Chat model exists:", chat_model is not None)
    st.write("Vector store exists:", vector_store is not None)
    
    # Temperature control
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    chat_model.temperature = temperature
    
    # Clear buttons
    if st.button("Clear Cache and Reload"):
        st.cache_resource.clear()
        st.experimental_rerun()
        
    if st.button("Clear Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me questions about the PDF document..."}
        ]
        memory.clear()
        st.experimental_rerun()

# Display file info
if os.path.exists("./data"):
    files = [f for f in os.listdir("./data") if f.endswith('.pdf')]
    if files:
        st.sidebar.success(f"📄 Loaded PDF: {files[0]}")

# Chat interface
prompt = st.chat_input("Your question")

# Display previous messages
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input and generate responses
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get response from the retrieval chain
                question = st.session_state.messages[-1]["content"]
                response = retrieval_chain({
                    "question": question,
                    "chat_history": [(msg["role"], msg["content"]) 
                                   for msg in st.session_state.messages 
                                   if msg["role"] != "assistant"]
                })
                
                # Display the assistant's response
                st.write(response['answer'])
                message = {"role": "assistant", "content": response['answer']}
                st.session_state.messages.append(message)

                # Display source documents in an expander
                if 'source_documents' in response:
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(response['source_documents']):
                            st.write(f"Source {i+1}:")
                            st.write(doc.page_content)
                            st.write("---")

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.exception(e)  # This will print the full traceback