import os
import streamlit as st
from dotenv import load_dotenv

# Updated imports for llama_index
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google.gemini import GeminiEmbedding

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate



st.set_page_config(page_title="Chat with Document", page_icon="📚")

Settings.llm = Gemini(
    model="models/gemini-pro",
    api_key="AIzaSyBFoHj9lZ4soXr0l-GDhg8TJO6PwRs0yoo"
)
Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001",
    api_key="AIzaSyBFoHj9lZ4soXr0l-GDhg8TJO6PwRs0yoo"
)


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner("Loading and indexing the documents..."):
        # Read and load documents from the "./data" directory
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        documents = reader.load_data()

        # Create a VectorStore index
        index = VectorStoreIndex.from_documents(documents)
        return index

# Load the index
index = load_data()

# Initialize chat engines and session state
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key="AIzaSyBFoHj9lZ4soXr0l-GDhg8TJO6PwRs0yoo",
        temperature=0.1
    )
    st.session_state.langchain_chat_engine = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory()
    )

# Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions about the document..."}
    ]

# Streamlit user interface
st.title("Chat With Document")
prompt = st.chat_input("Your question")

# Display previous messages
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input and generate responses
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        try:
            # Get context from document
            query_response = st.session_state.chat_engine.chat(prompt)
            context = query_response.response

            # If no relevant context is found, flag it
            if not context.strip():
                response = "The question is outside the scope of the provided context. Please contact the developer."
            else:
                # Format prompt with context
                template = ChatPromptTemplate.from_messages([
                    ("system", "Act as an AI assistant answering questions strictly based on the provided context. "
                            "If the question is irrelevant to the context, respond with: "
                            "'The question is outside the scope of the provided context. Please contact the developer.' "
                            f"Context: {context}"),
                    ("human", prompt),
                ])

                # Generate response
                # template.format_messages() might create structured objects. Extract content if needed.
                formatted_messages = template.format_messages(user_input=prompt)

                # Convert the formatted messages to plain strings
                formatted_messages = [msg.content if hasattr(msg, 'content') else str(msg) for msg in formatted_messages]

                # Generate response using the chat engine
                response = st.session_state.langchain_chat_engine.predict(input=formatted_messages)
                # Display the assistant's response
                response = response.replace("AIMessage(content='", "").replace("')", "")
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")