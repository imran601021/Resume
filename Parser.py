import pdfplumber
from google.colab import files

# Upload file from local system
uploaded = files.upload()

# Get the file name dynamically
file_name = list(uploaded.keys())[0]  # Get the uploaded file name
file_path = f"/content/{file_name}"  # Set file path

# ✅ Reverse the condition so it goes inside the block if it's NOT a PDF
if not file_name.lower().endswith(".pdf"):
    print("⚠️ File is not a PDF. Proceeding anyway...")
    resume_text = "This is a sample non-PDF file or fallback content."
    print(resume_text)  # Placeholder for non-PDF scenario
else:
    # Extract text if the file is a valid PDF
    def extract_text_from_pdf(file_path):
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Check if text is extracted from the page
                    text += page_text
        return text

    resume_text = extract_text_from_pdf(file_path)

    if resume_text.strip():
        print("✅ Resume text extracted successfully!")
        print(resume_text[:500])  # Print the first 500 characters
    else:
        print("⚠️ No text found in the PDF. Please check the file.")
