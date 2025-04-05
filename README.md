# Document Chat Assistant

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content using OpenAI's GPT models. The application provides a clean, modern interface for document interaction with real-time streaming responses.

## Features

- 📄 PDF document upload and processing
- 💬 Interactive chat interface with streaming responses
- 🔄 Real-time document processing
- 🔒 Secure API key management
- 🎨 Modern, responsive UI
- 📱 Mobile-friendly design

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Docker (optional, for containerized deployment)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd ask_doc
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t ask_doc .
```

2. Run the container:
```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=your_api_key_here ask_doc
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter your OpenAI API key in the application interface

4. Upload a PDF document

5. Start asking questions about the document's content

## Project Structure

```
ask_doc/
├── app.py              # Streamlit frontend application
├── backend.py          # Backend logic and API handling
├── document_helper.py  # Document processing utilities
├── static/
│   └── styles.css     # Custom CSS styles
├── temp/              # Temporary file storage
├── requirements.txt   # Python dependencies
└── Dockerfile         # Docker configuration
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `TEMP_UPLOAD_DIR`: Directory for temporary file storage (defaults to ./temp)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT models
- Streamlit for the web framework
- LangChain for the document processing capabilities 