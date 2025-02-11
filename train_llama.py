import sqlite3
import pandas as pd
import pickle
import os

def load_training_data_from_db(db_path):
    """
    Loads training data from the specified database.
    
    Assumes the database (researchers.db) has a table named 'works' with at least:
      - full_text
      - summary
      - summary_status = 'summarized'
      - progress = 1
    Returns a DataFrame with:
      - input_text (full text)
      - target_text (summary)
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT full_text, summary
        FROM works
        WHERE summary_status = 'summarized' AND progress = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df = df.rename(columns={"full_text": "input_text", "summary": "target_text"})
    return df

def process_and_save_data(processed_file, db_path):
    """
    Loads training data from the database and saves it to a pickle file.
    This file can be reloaded in subsequent training runs.
    """
    print("Loading training data from database...")
    df = load_training_data_from_db(db_path)
    with open(processed_file, "wb") as f:
        pickle.dump(df, f)
    print("Processed training data saved to file:", processed_file)
    return df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Relative path to the pickle file and DB
    processed_file = os.path.join(script_dir, "processed_training_data.pkl")
    db_path = os.path.join(script_dir, "researchers.db")
    
    df = process_and_save_data(processed_file, db_path)
    print(f"Retrieved {len(df)} training examples from the database.")
    
    from llama_model import fine_tune_llama_on_papers
    output_dir = fine_tune_llama_on_papers(df)
    print("Fine-tuned LLaMA model saved at:", output_dir)
