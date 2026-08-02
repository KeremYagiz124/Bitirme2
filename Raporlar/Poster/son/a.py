pip install pymupdf
python -c "import fitz; d=fitz.open('poster.pdf'); d[0].get_pixmap(dpi=300).save('poster.png')"