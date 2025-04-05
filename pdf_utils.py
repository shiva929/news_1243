import PyPDF2

def extract_pages(input_pdf, page_numbers, output_pdf):
    with open(input_pdf, "rb") as infile:
        reader = PyPDF2.PdfReader(infile)
        writer = PyPDF2.PdfWriter()

        for page_number in page_numbers:
            writer.add_page(reader.pages[page_number])  # 0-indexed

        with open(output_pdf, "wb") as outfile:
            writer.write(outfile)
    print(f"Pages {page_numbers} extracted to {output_pdf}")
