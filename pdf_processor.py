"""
PDF processing module for StudyBuddy.
Extracts text from uploaded PDF files, preserves document metadata (filename, page numbers),
and chunks text into optimal sizes for retrieval.

Supports both pypdf and pymupdf for maximum environment compatibility.
"""

from typing import List, Tuple
import io
import re

# Dual support: pypdf and pymupdf (fitz)
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pymupdf  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Simple standalone character text splitter fallback if langchain text splitters not installed
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=900, chunk_overlap=120, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_documents(self, docs: List[Document]) -> List[Document]:
                chunks = []
                for doc in docs:
                    text = doc.page_content
                    start = 0
                    while start < len(text):
                        end = min(start + self.chunk_size, len(text))
                        chunk_text = text[start:end]
                        chunks.append(Document(
                            page_content=chunk_text,
                            metadata=dict(doc.metadata)
                        ))
                        if end == len(text):
                            break
                        start += (self.chunk_size - self.chunk_overlap)
                return chunks


def extract_text_from_pdfs(uploaded_files) -> Tuple[List[Document], List[str]]:
    """
    Extracts text from a list of uploaded PDF files (or file-like objects).
    
    Returns:
        tuple: (raw_documents, warning_messages)
            - raw_documents: List of LangChain Document objects with page content and metadata.
            - warning_messages: List of human-friendly warnings for any unreadable or empty pages/files.
    """
    documents: List[Document] = []
    warnings: List[str] = []

    if not uploaded_files:
        return documents, ["No files provided for extraction."]

    if not HAS_PYPDF and not HAS_PYMUPDF:
        return documents, ["Neither 'pypdf' nor 'pymupdf' is installed. Please run: pip install pypdf"]

    # Support single file or list of files
    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    for file_obj in uploaded_files:
        filename = getattr(file_obj, "name", "uploaded_document.pdf")
        
        try:
            # Handle Streamlit UploadedFile or bytes
            file_bytes = None
            if hasattr(file_obj, "read"):
                file_bytes = file_obj.read()
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
            elif isinstance(file_obj, (bytes, bytearray)):
                file_bytes = bytes(file_obj)
            elif isinstance(file_obj, str):
                with open(file_obj, "rb") as f:
                    file_bytes = f.read()
            else:
                warnings.append(f"Skipping {filename}: Unsupported file format.")
                continue

            extracted_doc_count = 0

            # Method A: PyMuPDF if available
            if HAS_PYMUPDF:
                with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
                    num_pages = len(doc)
                    if num_pages == 0:
                        warnings.append(f"⚠️ {filename} appears to be empty (0 pages).")
                        continue

                    for page_idx, page in enumerate(doc, start=1):
                        text = page.get_text() or ""
                        cleaned_text = re.sub(r"\s+", " ", text).strip()
                        if cleaned_text:
                            documents.append(Document(
                                page_content=cleaned_text,
                                metadata={
                                    "source": filename,
                                    "page": page_idx,
                                    "total_pages": num_pages,
                                }
                            ))
                            extracted_doc_count += 1

            # Method B: pypdf fallback
            elif HAS_PYPDF:
                reader = PdfReader(io.BytesIO(file_bytes))
                num_pages = len(reader.pages)
                if num_pages == 0:
                    warnings.append(f"⚠️ {filename} appears to be empty (0 pages).")
                    continue

                for page_idx, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    cleaned_text = re.sub(r"\s+", " ", text).strip()
                    if cleaned_text:
                        documents.append(Document(
                            page_content=cleaned_text,
                            metadata={
                                "source": filename,
                                "page": page_idx,
                                "total_pages": num_pages,
                            }
                        ))
                        extracted_doc_count += 1

            if extracted_doc_count == 0:
                warnings.append(
                    f"⚠️ {filename} contained no readable text. It might be a scanned image or protected PDF."
                )

        except Exception as e:
            warnings.append(f"❌ Failed to parse '{filename}': {str(e)}")

    return documents, warnings


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 120
) -> List[Document]:
    """
    Splits document texts into smaller, overlapping chunks suitable for semantic search.
    Preserves original source and page metadata.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)
    
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks
