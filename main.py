#!/usr/bin/env python3
"""
Legal RAG System - Main Application
Interactive CLI for querying legal documents
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.rag_pipeline import LegalRAGPipeline


class LegalRAGApp:
    """Interactive application for Legal RAG system."""
    
    def __init__(
        self,
        collection_name: str = "legal_documents",
        embedder_model: str = "BAAI/bge-m3",
        llm_model: str = "llama3.1:8b"
    ):
        """Initialize the application."""
        print("🏛️  Legal RAG System")
        print("=" * 60)
        
        self.rag = LegalRAGPipeline(
            embedder_model=embedder_model,
            llm_model=llm_model,
            collection_name=collection_name
        )
    

    
    def interactive_query(self):
        """Start interactive query session."""
        print("\n" + "=" * 60)
        print("Interactive Query Mode")
        print("=" * 60)
        print("Commands:")
        print("  - Type your question to query the system")
        print("  - 'stats' - Show system statistics")
        print("  - 'help' - Show this help message")
        print("  - 'quit' or 'exit' - Exit the program")
        print("=" * 60 + "\n")
        
        while True:
            try:
                query = input("\n💬 Your question: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                elif query.lower() == 'help':
                    self._show_help()
                    continue
                
                elif query.lower() == 'stats':
                    self._show_stats()
                    continue
                
                # Process query
                print("\n🔍 Searching...")
                response = self.rag.query(
                    question=query,
                    top_k=5,
                    score_threshold=0.3,
                    return_sources=True
                )
                
                # Display answer
                print("\n" + "=" * 60)
                print("📋 Answer:")
                print("=" * 60)
                print(f"\n{response['answer']}\n")
                
                # Display sources
                if response.get('sources'):
                    print("=" * 60)
                    print(f"📚 Sources ({len(response['sources'])}):")
                    print("=" * 60)
                    
                    for i, source in enumerate(response['sources'][:3], 1):
                        print(f"\n{i}. Relevance: {source['score']:.3f}")
                        
                        if source['metadata']:
                            meta_str = ", ".join(
                                f"{k}: {v}" for k, v in source['metadata'].items()
                                if k not in ['chunk_index', 'total_chunks']
                            )
                            print(f"   📎 {meta_str}")
                        
                        # Show snippet
                        text_preview = source['text'][:200]
                        if len(source['text']) > 200:
                            text_preview += "..."
                        print(f"   📄 {text_preview}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        """Show help message."""
        print("\n" + "=" * 60)
        print("Help - Query Examples")
        print("=" * 60)
        print("""
Example queries:
- "What is the punishment for theft under IPC?"
- "Explain Article 21 of the Constitution"
- "What are the essential elements of a contract?"
- "Define defamation under Indian law"
- "What is the limitation period for filing a suit?"

Tips:
- Be specific in your questions
- Use legal terminology when applicable
- Ask one question at a time for best results
        """)
    
    def _show_stats(self):
        """Show system statistics."""
        stats = self.rag.get_stats()
        
        print("\n" + "=" * 60)
        print("System Statistics")
        print("=" * 60)
        print(f"\n📊 Vector Database:")
        print(f"   - Collection: {stats['vector_store']['name']}")
        print(f"   - Documents: {stats['vector_store']['points_count']}")
        print(f"   - Vector Size: {stats['vector_store']['vector_size']}D")
        print(f"\n🤖 Models:")
        print(f"   - Embedder: {stats['embedder']['model']}")
        print(f"   - Device: {stats['embedder']['device']}")
        print(f"   - LLM: {stats['llm']['model']}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Legal RAG System - Query legal documents using AI"
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        default='legal_documents',
        help='Qdrant collection name'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Single query (non-interactive mode)'
    )
    
    parser.add_argument(
        '--embedder',
        type=str,
        default='BAAI/bge-m3',
        help='Embedding model'
    )
    
    parser.add_argument(
        '--llm',
        type=str,
        default='llama3.1:8b',
        help='LLM model for Ollama'
    )
    
    args = parser.parse_args()
    
    # Initialize app
    try:
        app = LegalRAGApp(
            collection_name=args.collection,
            embedder_model=args.embedder,
            llm_model=args.llm
        )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check if model is pulled: ollama list")
        print("3. Install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    
    # Query mode
    if args.query:
        # Single query mode
        response = app.rag.query(args.query, return_sources=True)
        print(f"\n{'='*60}")
        print(f"Q: {args.query}")
        print(f"{'='*60}")
        print(f"\n{response['answer']}")
        
        if response.get('sources'):
            print(f"\n\nSources: {response['num_sources']}")
            for i, src in enumerate(response['sources'][:3], 1):
                print(f"{i}. [Score: {src['score']:.3f}] {src['text'][:100]}...")
    else:
        # Interactive mode
        app.interactive_query()


if __name__ == "__main__":
    main()