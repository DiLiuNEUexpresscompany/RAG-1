import os
import base64
import gc
import tempfile
import uuid
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

MODEL_NAME = "llama3.1"  


def get_llm(model_name):
    """Initialize LLM with specified model"""
    return Ollama(model=model_name, request_timeout=120.0)

def get_embeddings(model_name):
    """Initialize embedding model with specified model"""
    return OllamaEmbedding(
        model_name=model_name,
        request_timeout=120.0
    )

class DocumentChat:
    def __init__(self, model_name):
        self.initialize_session_state()
        self.model_name = model_name
        self.llm = get_llm(model_name)
        self.embed_model = get_embeddings(model_name)
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "id" not in st.session_state:
            st.session_state.id = uuid.uuid4()
            st.session_state.file_cache = {}
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def get_qa_prompt(self):
        """Return the custom QA prompt template"""
        template_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above, answer the query step by step in a concise manner. "
            "If you don't know the answer, say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        return PromptTemplate(template_str)
    
    def process_document(self, uploaded_file, temp_dir):
        """Process the uploaded document and create query engine"""
        try:
            loader = SimpleDirectoryReader(
                input_dir=temp_dir,
                required_exts=[".pdf"],
                recursive=True
            )
            docs = loader.load_data()
            
            # Manual text splitting logic (basic, can be enhanced)
            def simple_text_splitter(text, chunk_size=512, chunk_overlap=64):
                chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
                return chunks

            # Split the documents into chunks (simulating RecursiveCharacterTextSplitter)
            split_docs = [simple_text_splitter(doc.text) for doc in docs]
            
            # Create index using the global settings
            index = VectorStoreIndex.from_documents(
                docs,
                show_progress=True
            )
            
            # Create query engine with custom prompt
            query_engine = index.as_query_engine(
                streaming=True,
                similarity_top_k=3,
                text_qa_template=self.get_qa_prompt()
            )
            
            return query_engine
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return None
    
    @staticmethod
    def display_pdf(file):
        """Display the uploaded PDF"""
        try:
            st.markdown("### PDF Preview")
            base64_pdf = base64.b64encode(file.read()).decode("utf-8")
            pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>"""
            st.markdown(pdf_display, unsafe_allow_html=True)
            file.seek(0)  # Reset file pointer after reading
        except Exception as e:
            st.error(f"Error displaying PDF: {str(e)}")
    
    def handle_file_upload(self, uploaded_file):
        """Handle file upload and processing"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{st.session_state.id}-{uploaded_file.name}"
                
                if file_key not in st.session_state.file_cache:
                    st.info("Processing document... This may take a few minutes.")
                    progress_bar = st.progress(0)
                    
                    query_engine = self.process_document(uploaded_file, temp_dir)
                    if query_engine:
                        st.session_state.file_cache[file_key] = query_engine
                        progress_bar.progress(100)
                        st.success("Document processed successfully!")
                    else:
                        st.error("Failed to process document.")
                        return None
                
                self.display_pdf(uploaded_file)
                return file_key
                
        except Exception as e:
            st.error(f"An error occurred during file upload: {str(e)}")
            return None
    
    def handle_chat(self, prompt, file_key, use_rag):
        """Handle chat interaction and response generation"""
        try:
            if use_rag:
                # 需要上传文件的 RAG 模式
                if not file_key or file_key not in st.session_state.file_cache:
                    st.error("Please upload a document first.")
                    return None

                query_engine = st.session_state.file_cache[file_key]
                streaming_response = query_engine.query(prompt)

                full_response = ""
                message_placeholder = st.empty()

                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                return full_response
            else:
                # 普通对话模式，不需要文件
                message_placeholder = st.empty()
                response = self.llm.complete(prompt)
                
                # 如果 LLM 返回的是完整响应对象，需要获取文本内容
                response_text = response.text if hasattr(response, 'text') else str(response)
                
                # 显示响应
                message_placeholder.markdown(response_text)
                return response_text

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Chat with PDF",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar for model selection
    with st.sidebar:
        st.header("🤖 Model Configuration")
        selected_model = st.selectbox(
            "Choose Model",
            ["llama3.1", "gemma2:9b"],
            index=0,
            key="model_selector"
        )
        
        # RAG configuration
        use_rag = st.checkbox(
            "Enable RAG",
            value=st.session_state.get('use_rag', False),
            help="Enable or disable Retrieval-Augmented Generation",
            key="rag_checkbox"
        )
        
        # Update RAG state
        st.session_state.use_rag = use_rag
        
        # Handle model change
        if "current_model" not in st.session_state or st.session_state.current_model != selected_model:
            st.session_state.current_model = selected_model
            # Clear file cache when model changes
            if "file_cache" in st.session_state:
                st.session_state.file_cache = {}
            # Optionally clear chat history when model changes
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()

    # Initialize DocumentChat with current model
    doc_chat = DocumentChat(selected_model)

    # 文件上传部分
    with st.sidebar:
        if st.session_state.use_rag:
            st.header("📄 Document Upload (Required for RAG)")
            uploaded_file = st.file_uploader(
                "Upload your PDF file",
                type="pdf",
                help="Upload a PDF file to start chatting using RAG"
            )
        else:
            uploaded_file = None

        file_key = None
        if uploaded_file:
            file_key = doc_chat.handle_file_upload(uploaded_file)

    # Dynamic title with model name and RAG status
    title = f"💬 Chat with your PDF using {selected_model}"
    if st.session_state.use_rag:
        title += " (RAG Enabled)"
    st.header(title)

    # Clear chat button
    if st.button("Clear Chat History 🗑️"):
        st.session_state.messages = []
        st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Handle assistant response
        with st.chat_message("assistant"):
            response = doc_chat.handle_chat(prompt, file_key, st.session_state.use_rag)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()