from typing import List, Dict, Optional
from pathlib import Path
import re
from dataclasses import dataclass

# For PDF processing
try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# For DOCX processing
try:
    from docx import Document as DocxDocument
except ImportError:
    print("Install python-docx: pip install python-docx")


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    text: str
    metadata: Dict
    chunk_id: int
    source: str


class DocumentProcessor:
    """
    Process legal documents from various formats and chunk them.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size of each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF with table support."""
        if PDFPLUMBER_AVAILABLE:
            return self._load_pdf_with_pdfplumber(file_path)
        elif PYPDF_AVAILABLE:
            return self._load_pdf_with_pypdf(file_path)
        else:
            raise ImportError("No PDF library available. Install: pip install pdfplumber")
    
    def _load_pdf_with_pdfplumber(self, file_path: str) -> str:
        """Extract text and tables from PDF using pdfplumber."""
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text()
                if text and text.strip():
                    text_content.append(text)
                
                # Extract tables
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        # Convert table to structured text
                        table_text = "\n".join([
                            " | ".join(str(cell) if cell else "" for cell in row)
                            for row in table if any(cell for cell in row)
                        ])
                        if table_text.strip():
                            text_content.append(f"\n[TABLE {page_num}-{table_idx+1}]\n{table_text}\n")
        
        return "\n\n".join(text_content)
    
    def _load_pdf_with_pypdf(self, file_path: str) -> str:
        """Extract text from PDF using pypdf (fallback)."""
        import pypdf
        text_content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    text_content.append(text)
        
        return "\n\n".join(text_content)
    
    def load_docx(self, file_path: str) -> str:
        """Extract text from DOCX."""
        doc = DocxDocument(file_path)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text])
    
    def load_txt(self, file_path: str, encoding: str = 'utf-8') -> str:
        """Load text file."""
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    def load_document(self, file_path: str) -> str:
        """
        Load document from file (auto-detect format).
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text content
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.docx':
            return self.load_docx(file_path)
        elif suffix in ['.txt', '.md']:
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal citations
        # Keep: periods, commas, parentheses, forward slashes, hyphens
        text = re.sub(r'[^\w\s.,()\/\-:]', '', text)
        
        # Remove multiple consecutive periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences while respecting chunk size.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Split into sentences (simple regex for legal text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk_size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    # Keep last part for overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_section(self, text: str, section_pattern: str = None) -> List[str]:
        """
        Chunk text by legal sections/articles.
        
        Args:
            text: Text to chunk
            section_pattern: Regex pattern to identify sections
            
        Returns:
            List of text chunks
        """
        if section_pattern is None:
            # Default patterns for Indian legal documents
            section_pattern = r'(Section\s+\d+|Article\s+\d+|Chapter\s+[IVX]+)'
        
        # Split by sections
        sections = re.split(f'({section_pattern})', text, flags=re.IGNORECASE)
        
        chunks = []
        for i in range(1, len(sections), 2):
            if i+1 < len(sections):
                section_title = sections[i].strip()
                section_content = sections[i+1].strip()
                
                # If section is too large, further chunk it
                if len(section_content) > self.chunk_size * 1.5:
                    sub_chunks = self.chunk_by_sentences(section_content)
                    for sub_chunk in sub_chunks:
                        chunks.append(f"{section_title}: {sub_chunk}")
                else:
                    chunks.append(f"{section_title}: {section_content}")
        
        return chunks
    
    def process_document(
        self,
        file_path: str,
        metadata: Optional[Dict] = None,
        chunking_strategy: str = "sentences"
    ) -> List[DocumentChunk]:
        """
        Process a document into chunks.
        
        Args:
            file_path: Path to document
            metadata: Additional metadata for chunks
            chunking_strategy: 'sentences' or 'sections'
            
        Returns:
            List of DocumentChunk objects
        """
        # Load document
        text = self.load_document(file_path)
        
        # Clean text
        text = self.clean_text(text)
        
        # Chunk text
        if chunking_strategy == "sections":
            chunks = self.chunk_by_section(text)
        else:
            chunks = self.chunk_by_sentences(text)
        
        # Filter small chunks
        chunks = [c for c in chunks if len(c) >= self.min_chunk_size]
        
        # Create DocumentChunk objects
        doc_name = Path(file_path).stem
        base_metadata = metadata or {}
        
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "source_file": doc_name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            doc_chunk = DocumentChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=i,
                source=doc_name
            )
            document_chunks.append(doc_chunk)
        
        return document_chunks
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: List[str] = None
    ) -> List[DocumentChunk]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Path to directory
            recursive: Process subdirectories
            file_extensions: List of extensions to process
            
        Returns:
            List of all DocumentChunks
        """
        if file_extensions is None:
            file_extensions = ['.pdf', '.txt', '.docx', '.md']
        
        dir_path = Path(directory)
        all_chunks = []
        
        # Get all files
        if recursive:
            files = []
            for ext in file_extensions:
                files.extend(dir_path.rglob(f"*{ext}"))
        else:
            files = []
            for ext in file_extensions:
                files.extend(dir_path.glob(f"*{ext}"))
        
        print(f"Found {len(files)} documents to process")
        
        # Process each file
        for file_path in files:
            try:
                print(f"Processing: {file_path.name}")
                chunks = self.process_document(str(file_path))
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return all_chunks