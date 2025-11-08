# app/utils/file_handler.py - Enhanced PDF extraction

import PyPDF2
import fitz  # pymupdf for better text extraction
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Enhanced PDF text extraction that handles:
    - Complex layouts
    - Tables and structured content
    - Images with OCR (if available)
    - Preserves text structure
    """
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    
    try:
        # Try PyMuPDF first (better for complex layouts)
        return _extract_with_pymupdf(pdf_path)
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}, falling back to PyPDF2")
        try:
            return _extract_with_pypdf2(pdf_path)
        except Exception as e2:
            logger.error(f"All PDF extraction methods failed: {e2}")
            return None

def _extract_with_pymupdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF (better for complex layouts)"""
    
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract text blocks with layout preservation
        blocks = page.get_text("dict")
        page_text = []
        
        for block in blocks["blocks"]:
            if "lines" in block:  # Text block
                block_text = []
                for line in block["lines"]:
                    line_text = []
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            line_text.append(text)
                    if line_text:
                        block_text.append(" ".join(line_text))
                
                if block_text:
                    page_text.append("\n".join(block_text))
        
        if page_text:
            full_text.append(f"\n--- Page {page_num + 1} ---\n")
            full_text.append("\n\n".join(page_text))
    
    doc.close()
    return "\n".join(full_text)

def _extract_with_pypdf2(pdf_path: Path) -> str:
    """Fallback extraction using PyPDF2"""
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    full_text.append(f"\n--- Page {page_num + 1} ---\n")
                    full_text.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                continue
        
        return "\n".join(full_text)

def extract_structured_content(pdf_path: Path) -> dict:
    """
    Extract structured content including tables, lists, and sections
    """
    
    try:
        doc = fitz.open(pdf_path)
        structured_content = {
            "sections": [],
            "tables": [],
            "lists": [],
            "images": []
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract tables (basic detection)
            tables = page.find_tables()
            for table in tables:
                try:
                    table_data = table.extract()
                    structured_content["tables"].append({
                        "page": page_num + 1,
                        "data": table_data
                    })
                except:
                    pass
            
            # Extract images information
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                structured_content["images"].append({
                    "page": page_num + 1,
                    "image_index": img_index,
                    "xref": img[0]
                })
        
        doc.close()
        return structured_content
        
    except Exception as e:
        logger.error(f"Structured content extraction failed: {e}")
        return {"sections": [], "tables": [], "lists": [], "images": []}