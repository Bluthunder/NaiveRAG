"""
System Verification Test
Tests all components and checks if documents are ready for ingestion
"""

import sys
from pathlib import Path


def test_imports():
    """Test if all required modules can be imported."""
    print("="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        from src.embedder import DocumentEmbedder
        print("✓ embedder module")
        
        from src.vector_store import QdrantVectorStore
        print("✓ vector_store module")
        
        from src.llm_client import OllamaLLM
        print("✓ llm_client module")
        
        from src.rag_pipeline import LegalRAGPipeline
        print("✓ rag_pipeline module")
        
        from src.document_processing import DocumentProcessor
        print("✓ document_processor module")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_embedder():
    """Test embedder initialization."""
    print("\n" + "="*60)
    print("Testing Embedder...")
    print("="*60)
    
    try:
        from src.embedder import DocumentEmbedder
        
        embedder = DocumentEmbedder()
        print(f"✓ Model: {embedder.model_name}")
        print(f"✓ Device: {embedder.device}")
        print(f"✓ Embedding dimension: {embedder.embedding_dim}")
        
        # Test encoding
        test_text = "This is a test"
        embedding = embedder.encode_query(test_text)
        print(f"✓ Encoding works: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Embedder test failed: {e}")
        return False


def test_vector_store():
    """Test vector store initialization."""
    print("\n" + "="*60)
    print("Testing Vector Store...")
    print("="*60)
    
    try:
        from src.vector_store import QdrantVectorStore
        
        vs = QdrantVectorStore(
            collection_name="test_collection",
            embedding_dim=1024
        )
        print("✓ Qdrant connection successful")
        
        info = vs.get_collection_info()
        print(f"✓ Collection info: {info['points_count']} documents")
        
        return True
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        return False


def test_ollama():
    """Test Ollama LLM connection."""
    print("\n" + "="*60)
    print("Testing Ollama LLM...")
    print("="*60)
    
    try:
        from src.llm_client import OllamaLLM
        
        llm = OllamaLLM(model_name="llama3.1:8b")
        print(f"✓ Connected to Ollama")
        print(f"✓ Model: {llm.model_name}")
        
        # Quick test generation
        response = llm.generate("Say 'Hello'", max_tokens=10)
        print(f"✓ Generation works: {response[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Ollama test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print("  2. Check if model exists: ollama list")
        print("  3. Pull model if needed: ollama pull llama3.1:8b")
        return False


def check_data_folder():
    """Check if data folder exists and has documents."""
    print("\n" + "="*60)
    print("Checking data/ folder...")
    print("="*60)
    
    data_path = Path("data")
    
    if not data_path.exists():
        print("✗ data/ folder does not exist")
        print("\nAction required:")
        print("  mkdir data")
        return False
    
    print("✓ data/ folder exists")
    
    # Check for documents
    extensions = ['.pdf', '.docx', '.txt', '.md']
    files = []
    for ext in extensions:
        files.extend(list(data_path.rglob(f"*{ext}")))
    
    if not files:
        print("\n⚠ No documents found in data/ folder")
        print("\nAction required:")
        print("  1. Add your legal documents to data/ folder")
        print("  2. Supported formats: PDF, DOCX, TXT, MD")
        print("\nExample:")
        print("  cp /path/to/your/documents/*.pdf data/")
        return False
    
    print(f"\n✓ Found {len(files)} document(s):")
    for f in files[:5]:  # Show first 5
        print(f"  • {f.name}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")
    
    return True


def test_pipeline():
    """Test complete RAG pipeline."""
    print("\n" + "="*60)
    print("Testing RAG Pipeline...")
    print("="*60)
    
    try:
        from src.rag_pipeline import LegalRAGPipeline
        
        rag = LegalRAGPipeline(collection_name="test_pipeline")
        print("✓ Pipeline initialized successfully")
        
        # Test with a simple document
        test_docs = ["This is a test document about legal matters."]
        test_metadata = [{"source": "test", "type": "sample"}]
        
        rag.add_documents(test_docs, metadata=test_metadata)
        print("✓ Document addition works")
        
        # Test query
        response = rag.query("What is this about?", top_k=1)
        print(f"✓ Query works: {response['answer'][:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🏛️  Legal RAG System - Verification Test")
    
    results = {
        "Imports": test_imports(),
        "Embedder": False,
        "Vector Store": False,
        "Ollama LLM": False,
        "Data Folder": False,
        "RAG Pipeline": False
    }
    
    if results["Imports"]:
        results["Embedder"] = test_embedder()
        results["Vector Store"] = test_vector_store()
        results["Ollama LLM"] = test_ollama()
        results["Data Folder"] = check_data_folder()
        
        if all([results["Embedder"], results["Vector Store"], results["Ollama LLM"]]):
            results["RAG Pipeline"] = test_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour system is ready!")
        print("\nNext steps:")
        print("  1. Ensure documents are in data/ folder")
        print("  2. Run: python ingest_documents.py")
        print("  3. Run: python main.py")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the failed components above.")
        print("\nFor help, see:")
        print("  • QUICKSTART.md")
        print("  • SETUP_GUIDE.md")
        sys.exit(1)


if __name__ == "__main__":
    main()