from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from document_helper import db_handler
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import os
import openai
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    answer: str
    chat_history: List[tuple]

def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate OpenAI API key by making a test request.
    
    Args:
        api_key: OpenAI API key to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        # Create client with the API key
        client = OpenAI(api_key=api_key)
        
        # Make a test request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        return True, "API key is valid"
    except openai.AuthenticationError:
        return False, "Invalid API key"
    except openai.RateLimitError:
        return False, "Rate limit exceeded"
    except Exception as e:
        return False, f"Error validating API key: {str(e)}"

class DocumentChatBot:
    def __init__(self, api_key: Optional[str] = None):
        self.qa_chain = None
        self.api_key = api_key
        self.template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum. Keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 
        Chat History:
        {chat_history}

        Retrieved Context:
        {context}
        Question: {question}
        Helpful Answer:"""

    def set_api_key(self, api_key: str) -> Tuple[bool, str]:
        """
        Set and validate OpenAI API key.
        
        Args:
            api_key: OpenAI API key to set and validate
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        is_valid, message = validate_api_key(api_key)
        if is_valid:
            self.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        return is_valid, message

    def initialize_chain(self, file_path: str) -> None:
        """Initialize the QA chain with a new document"""
        if not self.api_key:
            raise ValueError("OpenAI API key not set. Please set it using set_api_key() method.")

        prompt = PromptTemplate(
            template=self.template,
            input_variables=["chat_history", "context", "question"]
        )

        # Create doc Retriever
        retriever = db_handler(file=file_path)

        # Initialize LLM
        llm = ChatOpenAI(temperature=0.0, api_key=self.api_key)

        # Initialize Memory
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            max_token_limit=2000
        )
        
        # Create ConversationalRetrievalChain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            combine_docs_chain_kwargs={"prompt": prompt},
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            return_generated_question=True,
        )

    def process_file(self, file_content: bytes, temp_path: str = None) -> bool:
        """Process an uploaded file"""
        try:
            # Use environment variable for temp path or fallback to local temp directory
            if temp_path is None:
                temp_dir = os.getenv('TEMP_UPLOAD_DIR', os.path.join(os.getcwd(), 'temp'))
                temp_path = os.path.join(temp_dir, 'temp.pdf')

            logger.info(f"Processing file, temp path: {temp_path}")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            logger.info(f"Created directory: {os.path.dirname(temp_path)}")
            
            # Save the file temporarily
            try:
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                logger.info(f"Successfully wrote file to: {temp_path}")
            except Exception as e:
                logger.error(f"Error writing file: {str(e)}")
                raise Exception(f"Failed to write file: {str(e)}")
            
            # Verify file was written correctly
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"File was not created at: {temp_path}")
            
            file_size = os.path.getsize(temp_path)
            logger.info(f"File size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("File is empty")
            
            # Initialize the QA chain
            try:
                self.initialize_chain(temp_path)
                logger.info("Successfully initialized QA chain")
                return True
            except Exception as e:
                logger.error(f"Error initializing QA chain: {str(e)}")
                raise Exception(f"Failed to initialize QA chain: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return False
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
                    pass

    def get_answer(self, question: str, chat_history: Optional[List[tuple]] = None) -> ChatResponse:
        """Get answer for a question"""
        if not self.qa_chain:
            return ChatResponse(
                answer="Please upload a document first!",
                chat_history=chat_history or []
            )

        if chat_history is None:
            chat_history = []

        try:
            response = self.qa_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })

            return ChatResponse(
                answer=response['answer'],
                chat_history=chat_history + [(question, response['answer'])]
            )
        except Exception as e:
            return ChatResponse(
                answer=f"Error processing question: {str(e)}",
                chat_history=chat_history
            ) 