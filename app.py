import streamlit as st
from backend import DocumentChatBot
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📚",
    layout="wide"
)

# Load and inject CSS
def load_css(css_file):
    with open(css_file) as f:
        return f.read()

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, "static", "styles.css")
st.markdown(f"<style>{load_css(css_path)}</style>", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_bot' not in st.session_state:
        st.session_state.chat_bot = DocumentChatBot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'question' not in st.session_state:
        st.session_state.question = ''
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''

def display_chat_messages():
    """Display chat messages"""
    # Display all messages
    for message in st.session_state.messages:
        if message[0]:  # User message
            st.markdown(
                f'<div class="chat-message user-message">{message[1]}</div>',
                unsafe_allow_html=True
            )
        else:  # Assistant message
            st.markdown(
                f'<div class="chat-message assistant-message">{message[1]}</div>',
                unsafe_allow_html=True
            )

def submit_question():
    if st.session_state.question.strip():
        try:
            # Store the current question
            current_question = st.session_state.question
            
            # Clear the input field immediately
            st.session_state.question = ''
            
            # Add user message
            st.session_state.messages.append((True, current_question))
            
            # Get the response
            response = st.session_state.chat_bot.get_answer(
                current_question,
                st.session_state.chat_history
            )
            
            # Add the response
            st.session_state.messages.append((False, response.answer))
            
            # Update chat history
            st.session_state.chat_history = response.chat_history
            
        except Exception as e:
            # Handle any errors
            error_message = f"Error processing question: {str(e)}"
            st.session_state.messages.append((False, error_message))

def handle_api_key_submit():
    """Handle API key submission"""
    api_key = st.session_state.api_key_input
    if api_key:
        is_valid, message = st.session_state.chat_bot.set_api_key(api_key)
        st.session_state.api_key_valid = is_valid
        st.session_state.api_key = api_key
        if is_valid:
            st.success(message)
        else:
            st.error(message)

def main():
    initialize_session_state()
    
    # Create two columns for title and clear button
    col1, col2 = st.columns([5, 1])
    
    with col1:
        st.title("📚 Document Chat Assistant")
    with col2:
        if st.session_state.uploaded_file and len(st.session_state.messages) > 0:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
    
    # API Key input using a form
    with st.form(key="api_key_form"):
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            key="api_key_input",
            value=st.session_state.api_key,
            help="Your API key will be stored securely in the session state"
        )
        submit_button = st.form_submit_button(
            "Submit API Key",
            on_click=handle_api_key_submit
        )

    if not st.session_state.api_key_valid:
        st.warning("Please enter a valid OpenAI API key to continue")
        return

    st.write("Upload a PDF document and ask questions about its content!")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
    
    if uploaded_file and uploaded_file != st.session_state.uploaded_file:
        with st.spinner("Processing document..."):
            try:
                if st.session_state.chat_bot.process_file(uploaded_file.getvalue()):
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.chat_history = []
                    st.session_state.messages = []
                    st.success("Document uploaded successfully! You can now ask questions about it.")
                else:
                    st.error("Error processing the document. Please try again.")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    # Chat interface
    if st.session_state.uploaded_file:
        # Create a container for the chat
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            display_chat_messages()

        # Question input
        st.text_input(
            "",
            key="question",
            on_change=submit_question,
            value=st.session_state.question,
            placeholder="Enter your question..."
        )

if __name__ == "__main__":
    main() 