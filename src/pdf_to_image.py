"""
Convert PDF to image for testing.
"""

import fitz  # PyMuPDF
from pathlib import Path

def convert_pdf_to_image(pdf_path, output_dir="data/raw"):
    """Convert first page of PDF to image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open PDF
    pdf = fitz.open(pdf_path)
    
    # Get first page
    page = pdf[0]
    
    # Convert to image
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    output_path = output_dir / f"{Path(pdf_path).stem}.png"
    pix.save(str(output_path))
    
    print(f"Image saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    pdf_path = "Reference Materials/Outline and Segments - 322 Owen Ave, Bessemer, AL 35020, USA.pdf"
    convert_pdf_to_image(pdf_path)
