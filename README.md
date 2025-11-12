# Legal RAG System 🏛️

A production-ready Retrieval-Augmented Generation (RAG) system for legal documents, running entirely locally using:
- **BAAI/bge-m3** for embeddings (1024-dimensional, optimized for legal text)
- **Llama 3.1 8B** via Ollama for LLM
- **Qdrant** as vector database

## ✨ Features

- 🔐 **Fully Local**: All processing happens on your machine - no data leaves your system
- 📚 **Multi-Format Support**: PDF, DOCX, TXT, Markdown
- 🎯 **Smart Chunking**: Sentence-based and section-based chunking strategies
- 🔍 **Semantic Search**: Using state-of-the-art BGE-M3 embeddings
- 💬 **Interactive CLI**: Easy-to-use command-line interface
- 📊 **Metadata Filtering**: Filter by source, section, or custom metadata
- ⚡ **GPU Accelerated**: Optional CUDA support for faster processing

## 🏗️ Architecture

```
┌─────────────┐
│  Documents  │
│ (PDF/DOCX)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│Document Processor│
│  (Chunking)     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  BAAI/bge-m3    │
│  (Embeddings)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│    Qdrant       │
│ (Vector Store)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────────┐
│  User Query     │─────▶│ Llama 3.1 8B │
│  (Retrieval)    │◀─────│   (Ollama)   │
└─────────────────┘      └──────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd LEGALRAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install . 
```

### 2. Install Ollama and Pull Model

```bash
# Install Ollama (https://ollama.com)
# Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3.1 8B
ollama pull llama3.1:8b

# Start Ollama server (in separate terminal)
ollama serve
```

### 3. Add Documents to data/ Folder

```bash
# Place your legal documents in the data/ folder
# Supported formats: PDF, DOCX, TXT, MD
cp /path/to/your/documents/*.pdf data/
```

### 4. Ingest Documents

```bash
# Process and embed all documents from data/ folder
uv run python ingest_documents.py
```

### 5. Query the System

```bash
# Interactive mode
uv run python main.py

# Single query
uv run python main.py --query "What is the punishment for theft?"
```

## 📖 Usage Examples

### Interactive Mode

```bash
$ uv run python main.py

🏛️  Legal RAG System
============================================================
...
Interactive Query Mode
============================================================

💬 Your question: What are the penalties for breach of contract?

🔍 Searching...

============================================================
📋 Answer:
============================================================

Under Section 73 of the Indian Contract Act, 1872, when a contract 
has been broken, the party who suffers by such breach is entitled to 
receive compensation for any loss or damage caused to him thereby...

============================================================
📚 Sources (3):
============================================================

1. Relevance: 0.847
   📎 source: Contract Act, section: 73
   📄 Section 73 of the Indian Contract Act deals with compensation...
```

### Programmatic Usage

### Batch Processing

```python
# Use the ingestion script for batch processing
# Command line:
uv run python ingest_documents.py --data-dir data/ --chunk-size 500

# Or programmatically:
from src.document_processor import DocumentProcessor
from src.rag_pipeline import LegalRAGPipeline

processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
rag = LegalRAGPipeline()

# Process directory
chunks = processor.process_directory("data/legal_docs", recursive=True)

# Add to RAG
texts = [c.text for c in chunks]
metadata = [c.metadata for c in chunks]
rag.add_documents(texts, metadata=metadata)
```

## 🎯 Configuration

### Chunk Size Optimization

```python
processor = DocumentProcessor(
    chunk_size=500,      # Characters per chunk (400-600 for legal docs)
    chunk_overlap=50,    # Overlap to maintain context
    min_chunk_size=100   # Filter out very small chunks
)
```

### Embedding Parameters

```python
embedder = DocumentEmbedder(
    model_name="BAAI/bge-m3",
    device="cuda",        # or "cpu"
    normalize=True        # Normalize embeddings
)
```

### LLM Parameters

```python
llm = OllamaLLM(
    model_name="llama3.1:8b",
    temperature=0.3,      # 0.0-1.0 (lower = more focused)
    top_p=0.9,
    max_tokens=2048
)
```

### Retrieval Parameters

```python
response = rag.query(
    question="Your question",
    top_k=5,              # Number of documents to retrieve
    score_threshold=0.5,  # Minimum similarity score (0-1)
    filters={             # Optional metadata filters
        "source": "",
        "topic": "fraud"
    }
)
```

## 📊 Performance

### Benchmarks (Approximate)

| Component | Operation | Time | Hardware |
|-----------|-----------|------|----------|
| Embeddings | 1000 chunks | ~30s | RTX 3060 |
| Embeddings | 1000 chunks | ~3min | CPU (Intel i7) |
| Vector Search | Query | <100ms | Local Qdrant |
| LLM Generation | Answer | 2-5s | Ollama (8B model) |

### Optimization Tips

1. **Use GPU**: 10-50x faster for embeddings
2. **Batch Processing**: Process documents in batches of 32-64
3. **Quantized Models**: Use `llama3.1:8b-q4` for faster inference
4. **Chunk Size**: 400-600 chars optimal for legal documents
5. **Caching**: Qdrant automatically caches frequent queries

## 🔧 Advanced Features

### Custom Chunking Strategy

```python
# Section-based chunking (for structured legal documents)
chunks = processor.process_document(
    "ipc.pdf",
    chunking_strategy="sections"  # Splits by Section/Article markers
)

# Sentence-based chunking (for general text)
chunks = processor.process_document(
    "judgment.pdf",
    chunking_strategy="sentences"
)
```

### Metadata Filtering

```python
# Query specific sources


# Query by date range

```

### Citation Tracking

```python
response = rag.query("Your question", return_sources=True)

for i, source in enumerate(response['sources'], 1):
    print(f"Citation {i}:")
    print(f"  Text: {source['text']}")
    print(f"  Source: {source['metadata']['source']}")
    print(f"  Confidence: {source['score']:.2%}")
```

## 📁 Project Structure Details

```
LEGALRAG/
├── src/
│   ├── __init__.py
│   ├── embedder.py           # Embedding generation using BGE-M3
│   ├── vector_store.py       # Qdrant vector database interface
│   ├── llm_client.py         # Ollama API client for Llama 3.1
│   ├── rag_pipeline.py       # Main RAG orchestration
│   └── document_processor.py # Document loading and chunking
├── data/                     # Your legal documents (place here)
├── qdrant_data/             # Vector database storage (auto-created)
├── ingest_documents.py      # Document ingestion script
├── main.py                  # Query interface (CLI)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🐛 Troubleshooting

### Ollama Not Running

```
❌ Error: Could not connect to Ollama
```

**Solution:**
```bash
ollama serve
# In another terminal:
ollama list  # Verify model is available
```

### CUDA Out of Memory

```
❌ Error: CUDA out of memory
```

**Solutions:**
- Reduce batch size: `batch_size=8` or `16`
- Use CPU: `device="cpu"`
- Use smaller model: `llama3.1:8b-q4`

### Model Not Found

```
⚠️ Model llama3.1:8b not found
```

**Solution:**
```bash
ollama pull llama3.1:8b
```

### Slow Performance

**For Embeddings:**
- Use GPU if available
- Increase batch size (if GPU memory allows)
- Cache embeddings for repeated documents

**For LLM:**
- Use quantized model (`llama3.1:8b-q4`)
- Reduce `max_tokens` parameter
- Consider using smaller model

## 🔐 Security & Privacy

- ✅ **100% Local**: No data sent to external servers
- ✅ **Private**: All documents and queries stay on your machine
- ✅ **Secure**: No API keys or cloud services required
- ✅ **Offline**: Works completely offline after initial model downloads

## 📚 Supported Document Types

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Native support via pypdf |
| Word | `.docx` | Native support via python-docx |
| Text | `.txt` | UTF-8 encoding |
| Markdown | `.md` | Treated as plain text |

## 🎓 Use Cases

1. **Legal Research**: Query across case laws, statutes, and legal documents
2. **Contract Analysis**: Find relevant clauses and precedents
3. **Compliance**: Search regulations and compliance requirements
4. **Legal Education**: Study aid for law students
5. **Document Review**: Quick reference for legal professionals

## 🛠️ Extending the System

### Add New Document Types

```python
# In document_processor.py
def load_custom_format(self, file_path: str) -> str:
    # Your custom loading logic
    return extracted_text
```

### Custom Prompts

```python
# In llm_client.py, modify RAGLLMClient
system_prompt = """
Your custom system prompt for specific legal domains
"""
```

### Integration with Web Interface

```python
from flask import Flask, request, jsonify
from src.rag_pipeline import LegalRAGPipeline

app = Flask(__name__)
rag = LegalRAGPipeline()

@app.route('/query', methods=['POST'])
def query():
    question = request.json['question']
    response = rag.query(question)
    return jsonify(response)
```

## 📈 Roadmap

- [ ] Web interface (Gradio/Streamlit)
- [ ] Support for more legal document formats (XML, JSON)
- [ ] Multi-language support (Hindi, other Indian languages)
- [ ] Fine-tuned legal domain embeddings
- [ ] Case law citation network analysis
- [ ] Export to legal citation formats

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Legal domain-specific fine-tuning
- Additional document formats
- Performance optimizations
- UI/UX improvements

## 📄 License

This project is for educational and research purposes. Ensure you have proper rights to process any legal documents you use with this system.

## 🙏 Acknowledgments

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for BGE-M3 embeddings
- [Ollama](https://ollama.com) for local LLM deployment
- [Qdrant](https://qdrant.tech) for vector database
- [Sentence Transformers](https://www.sbert.net/) for embedding framework

## 📞 Support

For issues and questions:
- Check the [troubleshooting section](#-troubleshooting)
- Review Ollama docs: https://ollama.com/docs
- Review Qdrant docs: https://qdrant.tech/documentation/

---

**Note**: This system provides information retrieval and is not a substitute for professional legal advice. Always consult with qualified legal professionals for legal matters.