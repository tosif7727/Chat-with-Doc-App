"""
Chat with Document App
A Streamlit application that allows users to upload documents and chat with them using ChatGPT API.
Features: Document processing, intelligent caching, conversation history management.
"""

import streamlit as st
import openai
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re

# Document processing libraries
try:
    import PyPDF2
    from docx import Document
except ImportError:
    st.error("Required libraries not installed. Please install: pypdf2, python-docx")

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Chat with Document",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHE MANAGEMENT CLASS
# ============================================================================

class CacheManager:
    """Manages caching for API responses and document processing"""
    
    def __init__(self, cache_dir: str = ".cache", max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.response_cache_file = self.cache_dir / "responses.json"
        self.document_cache_file = self.cache_dir / "documents.json"
        
        # Initialize cache files if they don't exist
        for cache_file in [self.response_cache_file, self.document_cache_file]:
            if not cache_file.exists():
                cache_file.write_text("{}")
    
    def _generate_key(self, data: str) -> str:
        """Generate a hash key for caching"""
        return hashlib.md5(data.encode()).hexdigest()
    
    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache from file"""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading cache: {e}")
            return {}
    
    def _save_cache(self, cache_file: Path, cache_data: Dict):
        """Save cache to file"""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"Error saving cache: {e}")
    
    def _is_expired(self, timestamp: str) -> bool:
        """Check if cache entry is expired"""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            return datetime.now() - cached_time > self.max_age
        except:
            return True
    
    def get_response(self, prompt: str, context: str) -> Optional[str]:
        """Get cached API response"""
        cache = self._load_cache(self.response_cache_file)
        key = self._generate_key(f"{prompt}|{context}")
        
        if key in cache:
            entry = cache[key]
            if not self._is_expired(entry.get('timestamp', '')):
                return entry.get('response')
        return None
    
    def set_response(self, prompt: str, context: str, response: str):
        """Cache API response"""
        cache = self._load_cache(self.response_cache_file)
        key = self._generate_key(f"{prompt}|{context}")
        
        cache[key] = {
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt[:100]  # Store truncated prompt for reference
        }
        
        self._save_cache(self.response_cache_file, cache)
    
    def get_document(self, file_hash: str) -> Optional[str]:
        """Get cached document content"""
        cache = self._load_cache(self.document_cache_file)
        
        if file_hash in cache:
            entry = cache[file_hash]
            if not self._is_expired(entry.get('timestamp', '')):
                return entry.get('content')
        return None
    
    def set_document(self, file_hash: str, content: str, filename: str):
        """Cache document content"""
        cache = self._load_cache(self.document_cache_file)
        
        cache[file_hash] = {
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'filename': filename
        }
        
        self._save_cache(self.document_cache_file, cache)
    
    def clear_cache(self):
        """Clear all cache"""
        for cache_file in [self.response_cache_file, self.document_cache_file]:
            cache_file.write_text("{}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        response_cache = self._load_cache(self.response_cache_file)
        document_cache = self._load_cache(self.document_cache_file)
        
        return {
            'response_count': len(response_cache),
            'document_count': len(document_cache),
            'cache_size_kb': sum(
                f.stat().st_size for f in self.cache_dir.glob('*.json')
            ) / 1024
        }

# ============================================================================
# DOCUMENT PROCESSOR CLASS
# ============================================================================

class DocumentProcessor:
    """Handles document upload and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8').strip()
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""
    
    @staticmethod
    def get_file_hash(file_content: bytes) -> str:
        """Generate hash for file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    def process_document(self, uploaded_file, cache_manager: CacheManager) -> str:
        """Process uploaded document and extract text"""
        # Read file content
        file_content = uploaded_file.read()
        file_hash = self.get_file_hash(file_content)
        
        # Check cache first
        cached_content = cache_manager.get_document(file_hash)
        if cached_content:
            st.success("📦 Loaded from cache!")
            return cached_content
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Extract text based on file type
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.pdf':
            text = self.extract_text_from_pdf(uploaded_file)
        elif file_extension == '.docx':
            text = self.extract_text_from_docx(uploaded_file)
        elif file_extension == '.txt':
            text = self.extract_text_from_txt(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
        
        # Cache the extracted text
        if text:
            cache_manager.set_document(file_hash, text, uploaded_file.name)
        
        return text

# ============================================================================
# CHATGPT INTEGRATION
# ============================================================================

class ChatGPTHandler:
    """Handles ChatGPT API interactions"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
    
    def chunk_text(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        """Split text into chunks for context management"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1
            if current_size > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_relevant_context(self, question: str, document_text: str, max_context_size: int = 3000) -> str:
        """Extract relevant context from document based on question"""
        # Simple keyword-based relevance (can be improved with embeddings)
        chunks = self.chunk_text(document_text, max_chunk_size=1000)
        
        # Score chunks based on keyword overlap
        question_words = set(re.findall(r'\w+', question.lower()))
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(re.findall(r'\w+', chunk.lower()))
            score = len(question_words & chunk_words)
            scored_chunks.append((score, chunk))
        
        # Sort by score and combine top chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        
        context = ""
        for score, chunk in scored_chunks:
            if len(context) + len(chunk) <= max_context_size:
                context += chunk + "\n\n"
            else:
                break
        
        return context.strip() if context else document_text[:max_context_size]
    
    def chat(self, question: str, document_text: str, conversation_history: List[Dict], 
             cache_manager: CacheManager) -> str:
        """Send question to ChatGPT with document context"""
        
        # Get relevant context
        context = self.get_relevant_context(question, document_text)
        
        # Check cache first
        cached_response = cache_manager.get_response(question, context)
        if cached_response:
            st.info("💾 Response loaded from cache")
            return cached_response
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant that answers questions based on the provided document. 
                Use the following document context to answer questions accurately and concisely.
                If the answer is not in the document, say so clearly.
                
                Document Context:
                {context}
                """
            }
        ]
        
        # Add conversation history (last 5 exchanges)
        for msg in conversation_history[-10:]:
            messages.append(msg)
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        try:
            # Call ChatGPT API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Cache the response
            cache_manager.set_response(question, context, answer)
            
            return answer
            
        except Exception as e:
            st.error(f"Error calling ChatGPT API: {e}")
            return "Sorry, I encountered an error processing your request."

# ============================================================================
# STREAMLIT UI
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    if 'document_name' not in st.session_state:
        st.session_state.document_name = ""

def render_sidebar():
    """Render sidebar with settings and information"""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="Select the ChatGPT model to use"
        )
        
        st.divider()
        
        # Cache management
        st.subheader("📦 Cache Management")
        
        cache_stats = st.session_state.cache_manager.get_cache_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Responses", cache_stats['response_count'])
        with col2:
            st.metric("Documents", cache_stats['document_count'])
        
        st.caption(f"Cache Size: {cache_stats['cache_size_kb']:.2f} KB")
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.cache_manager.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
        
        st.divider()
        
        # Conversation management
        st.subheader("💬 Conversation")
        
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.success("Started new conversation!")
            st.rerun()
        
        st.caption(f"Messages: {len(st.session_state.conversation_history)}")
        
        st.divider()
        
        # Information
        st.subheader("ℹ️ About")
        st.markdown("""
        **Chat with Document** allows you to:
        - Upload PDF, DOCX, or TXT files
        - Ask questions about the content
        - Get AI-powered answers
        - Benefit from intelligent caching
        
        **Features:**
        - 🚀 Fast response with caching
        - 💾 Document caching
        - 🔄 Conversation history
        - 🎯 Context-aware answers
        """)
        
        return api_key, model

def render_main_content(api_key: str, model: str):
    """Render main content area"""
    
    st.title("📚 Chat with Document")
    st.markdown("Upload a document and start asking questions!")
    
    # Document upload section
    st.subheader("📄 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_file:
        # Process document
        processor = DocumentProcessor()
        
        with st.spinner("Processing document..."):
            document_text = processor.process_document(
                uploaded_file,
                st.session_state.cache_manager
            )
        
        if document_text:
            st.session_state.document_text = document_text
            st.session_state.document_name = uploaded_file.name
            
            # Show document info
            st.success(f"✅ Document loaded: **{uploaded_file.name}**")
            
            with st.expander("📖 Document Preview"):
                st.text_area(
                    "Content",
                    document_text[:1000] + "..." if len(document_text) > 1000 else document_text,
                    height=200,
                    disabled=True
                )
                st.caption(f"Total characters: {len(document_text):,}")
    
    st.divider()
    
    # Chat interface
    if st.session_state.document_text:
        st.subheader("💬 Chat")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("### Conversation History")
            
            for msg in st.session_state.conversation_history:
                role = msg['role']
                content = msg['content']
                
                if role == 'user':
                    st.markdown(f"**🧑 You:** {content}")
                else:
                    st.markdown(f"**🤖 Assistant:** {content}")
                st.markdown("---")
        
        # Question input
        question = st.text_input(
            "Ask a question about the document",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🚀 Ask", use_container_width=True)
        
        if ask_button and question:
            if not api_key:
                st.error("⚠️ Please enter your OpenAI API key in the sidebar")
            else:
                # Initialize ChatGPT handler
                chatgpt = ChatGPTHandler(api_key, model)
                
                # Get response
                with st.spinner("Thinking..."):
                    answer = chatgpt.chat(
                        question,
                        st.session_state.document_text,
                        st.session_state.conversation_history,
                        st.session_state.cache_manager
                    )
                
                # Update conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": question
                })
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer
                })
                
                # Rerun to update UI
                st.rerun()
    
    else:
        st.info("👆 Please upload a document to start chatting")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    initialize_session_state()
    
    # Render UI
    api_key, model = render_sidebar()
    render_main_content(api_key, model)
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit & OpenAI • Chat with Document App v1.0")

if __name__ == "__main__":
    main()
