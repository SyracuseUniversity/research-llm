import os
import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths (adjust if needed)
FINE_TUNED_MODEL_PATH = r"C:\codes\llama32\fine_tuned_llama"
DB_PATH = r"C:\codes\t5-db\researchers.db"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned model and tokenizer
def load_fine_tuned_model():
    print(f"Loading fine-tuned model from: {FINE_TUNED_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model and tokenizer loaded.")
    return model, tokenizer

# Load database connection
def connect_database():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    print(f"✅ Connected to database at: {DB_PATH}")
    return conn

# Search database for matching papers or researchers
def search_database(conn, keyword):
    cursor = conn.cursor()
    query = """
        SELECT researcher_name, work_title, authors, info
        FROM research_info
        WHERE researcher_name LIKE ? OR work_title LIKE ? OR authors LIKE ? OR info LIKE ?
        LIMIT 5
    """
    params = (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
    cursor.execute(query, params)
    results = cursor.fetchall()
    return results

# Generate chatbot answer using LLaMA
def generate_llama_answer(model, tokenizer, question):
    prompt = (
        "You are a Syracuse University research assistant trained to answer questions about Syracuse-affiliated authors, papers, and subjects.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in generated_text:
        return generated_text.split("Answer:")[-1].strip()
    else:
        return generated_text.strip()

# Chatbot loop
def run_chatbot():
    model, tokenizer = load_fine_tuned_model()
    conn = connect_database()

    print("\n🤖 Chatbot is ready! Type your research question or keyword (or 'exit' to quit):\n")
    greetings = {"hello", "hi", "hey"}

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting chatbot.")
            break

        if question.lower() in greetings:
            print("Chatbot: Hello! How can I help you with your research questions today?")
            continue

        # First, search database for quick matches
        matches = search_database(conn, question)
        if matches:
            print("📚 Found related research entries:")
            for idx, (researcher_name, work_title, authors, info) in enumerate(matches, 1):
                print(f" {idx}. {work_title} by {authors}\n    Info: {info}\n")
        
        # Then generate LLaMA answer
        answer = generate_llama_answer(model, tokenizer, question)
        print("Chatbot:", answer)
    
    conn.close()
    print("🔒 Database connection closed.")

if __name__ == "__main__":
    run_chatbot()
