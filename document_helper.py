from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
from langchain.chains.question_answering import load_qa_chain
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def db_handler(file: str, k: int = 4) -> DocArrayInMemorySearch:
    """
    Process PDF document and create vector storage.
    
    Args:
        file: Path to the PDF file
        k: Number of documents to retrieve for each query
        
    Returns:
        Retriever object for document search
    """
    try:
        logger.info(f"Starting to process file: {file}")
        
        # Verify file exists and is readable
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
            
        # Load and process the PDF
        try:
            loader = PyPDFLoader(file)
            documents = loader.load()
            logger.info(f"Successfully loaded PDF with {len(documents)} pages")
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise Exception(f"Failed to load PDF: {str(e)}")

        # Split the document into chunks
        try:
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=30,
                separator="\n"
            )
            docs = text_splitter.split_documents(documents=documents)
            logger.info(f"Split documents into {len(docs)} chunks")
        except Exception as e:
            logger.error(f"Error splitting document: {str(e)}")
            raise Exception(f"Failed to process document chunks: {str(e)}")

        # Create vector embeddings
        try:
            embeddings = OpenAIEmbeddings()
            vector_store = DocArrayInMemorySearch.from_documents(docs, embeddings)
            logger.info("Successfully created vector embeddings")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")

        # Create and return retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        logger.info("Successfully created retriever")
        
        return retriever
        
    except Exception as e:
        logger.error(f"Error in db_handler: {str(e)}")
        raise Exception(f"Document processing failed: {str(e)}")


def chat_with_document(qa_chain: ConversationalRetrievalChain, query: str, chat_history: List[tuple] = None) -> Dict[str, Any]:
    """
    Handle a single chat interaction with the document.
    
    Args:
        qa_chain: The ConversationalRetrievalChain instance
        query: The user's question
        chat_history: List of tuples containing (human_message, ai_message)
        
    Returns:
        Dictionary containing the response and updated chat history
    """
    if chat_history is None:
        chat_history = []
    
    response = qa_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })
    
    # Extract the answer
    answer = response['answer']
    
    # Update chat history with the new interaction
    chat_history.append((query, answer))
    
    return {
        "answer": answer,
        "chat_history": chat_history
    }


def interactive_chat_session(qa_chain: ConversationalRetrievalChain):
    """
    Start an interactive chat session with the document.
    
    Args:
        qa_chain: The ConversationalRetrievalChain instance
    """
    chat_history = []
    print("Welcome to the document chat! Type 'quit' to exit.")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'quit':
            break
            
        response = chat_with_document(qa_chain, query, chat_history)
        print("\nAnswer:", response['answer'])
        chat_history = response['chat_history']


if __name__ == "__main__":
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 
    Chat History:
    {chat_history}

    Retrieved Context:
    {context}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(template=template,
        input_variables=["chat_history", "context", "question"])

    # Create doc Retriever
    retriever = db_handler(file="ReAct.pdf")

    # Initialize LLM
    llm = ChatOpenAI(temperature=0.0)

    # Initialize Memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        max_token_limit=2000
    )

    
    # Create ConversationalRetrievalChain with Custom Prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        combine_docs_chain_kwargs={"prompt": prompt}, 
        llm=llm,
        retriever=retriever,
        memory=memory,
        #verbose=True,
        return_source_documents=True,
        return_generated_question=True,
    )

    # Start interactive chat session
    interactive_chat_session(qa_chain)
    
    
