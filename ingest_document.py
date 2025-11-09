
"""
Document Ingestion Script
Processes documents from data/ folder and adds them to the vector database
"""

import argparse
from pathlib import Path
from typing import List
import sys

from src.document_processing import DocumentProcessor
from src.rag_pipeline import LegalRAGPipeline


def ingest_documents(
    data_dir: str = "data",
    collection_name: str = "legal_documents",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: str = "sentences",
    recursive: bool = True,
    file_extensions: List[str] = None
):
    """
    Ingest all documents from data directory into the vector database.
    
    Args:
        data_dir: Directory containing documents
        collection_name: Qdrant collection name
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        chunking_strategy: 'sentences' or 'sections'
        recursive: Process subdirectories
        file_extensions: List of file extensions to process
    """
    
    print("="*70)
    print("LEGAL RAG - DOCUMENT INGESTION")
    print("="*70)
    
    # Check if data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\n❌ Error: Directory '{data_dir}' does not exist")
        print(f"Please create it and add your legal documents")
        sys.exit(1)
    
    # Initialize components
    print("\n📦 Initializing components...")
    try:
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        rag = LegalRAGPipeline(
            collection_name=collection_name,
            qdrant_path="./qdrant_data"
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Process documents
    print(f"\n📂 Scanning directory: {data_dir}")
    print(f"   Chunking strategy: {chunking_strategy}")
    print(f"   Chunk size: {chunk_size} chars (overlap: {chunk_overlap})")
    
    try:
        chunks = processor.process_directory(
            directory=data_dir,
            recursive=recursive,
            file_extensions=file_extensions
        )
    except Exception as e:
        print(f"\n❌ Error processing documents: {e}")
        sys.exit(1)
    
    if not chunks:
        print(f"\n⚠️  No documents found in '{data_dir}'")
        print(f"\nSupported formats: PDF, DOCX, TXT, MD")
        print(f"Please add documents to the data/ folder")
        sys.exit(0)
    
    print(f"\n✅ Processed {len(chunks)} chunks from documents")
    
    # Display summary by source
    source_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get('source_file', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n📊 Document Summary:")
    for source, count in sorted(source_counts.items()):
        print(f"   • {source}: {count} chunks")
    
    # Add to vector database
    print(f"\n💾 Adding to vector database...")
    try:
        texts = [chunk.text for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        
        doc_ids = rag.add_documents(
            documents=texts,
            metadata=metadata,
            batch_size=32,
            show_progress=True
        )
        
        print(f"\n✅ Successfully ingested {len(doc_ids)} chunks")
        
        # Show statistics
        stats = rag.get_stats()
        print(f"\n📈 Database Statistics:")
        print(f"   Collection: {stats['vector_store']['name']}")
        print(f"   Total documents: {stats['vector_store']['points_count']}")
        print(f"   Vector dimension: {stats['vector_store']['vector_size']}D")
        
    except Exception as e:
        print(f"\n❌ Error adding to database: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ INGESTION COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext step: Query your documents")
    print("   python main.py")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest legal documents into the RAG system"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing documents (default: data)'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        default='legal_documents',
        help='Qdrant collection name (default: legal_documents)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Chunk size in characters (default: 500)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Overlap between chunks (default: 50)'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['sentences', 'sections'],
        default='sentences',
        help='Chunking strategy (default: sentences)'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.pdf', '.docx', '.txt', '.md'],
        help='File extensions to process (default: .pdf .docx .txt .md)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not process subdirectories'
    )
    
    args = parser.parse_args()
    
    ingest_documents(
        data_dir=args.data_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.strategy,
        recursive=not args.no_recursive,
        file_extensions=args.extensions
    )


if __name__ == "__main__":
    main()