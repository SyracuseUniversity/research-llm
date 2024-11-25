# main.py

from summarize import load_trained_model, summarize_text
from preprocessing import extract_text_from_pdf, preprocess_text

def main():
    pdf_path = r"C:\codes\test_training\train_pdfs\jones_et_al_2022.pdf"

    # Extract and preprocess text from the PDF
    print(f"\nProcessing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if text:
        cleaned_text = preprocess_text(text)

        # Load the trained model
        tokenizer, model = load_trained_model()

        # Generate the summary
        print("\nGenerating summary...")
        summary = summarize_text(cleaned_text, tokenizer, model)
        print("\nSummary:")
        print(summary)
    else:
        print(f"No text found in {pdf_path}.")

if __name__ == '__main__':
    main()
