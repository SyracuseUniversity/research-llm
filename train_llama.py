# # train_llama.py
# import sqlite3
# import pandas as pd
# import pickle

# def load_training_data_from_db(db_path):
#     """
#     Loads training data from the specified database.
    
#     Assumes the database (researchers.db) has a table named 'works' with at least:
#       - full_text: the full text of the paper.
#       - summary: the T5-generated summary.
#       - summary_status: should be 'summarized'.
#       - progress: should be 1.
    
#     Returns a DataFrame with:
#       - input_text: full paper text.
#       - target_text: summary.
#     """
#     conn = sqlite3.connect(db_path)
#     query = """
#         SELECT full_text, summary 
#         FROM works
#         WHERE summary_status = 'summarized' AND progress = 1
#     """
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     df = df.rename(columns={"full_text": "input_text", "summary": "target_text"})
#     return df

# def process_and_save_data(processed_file, db_path):
#     """
#     Loads training data from the database and saves it to a pickle file.
#     This file can be reloaded in subsequent training runs.
#     """
#     print("Loading training data from database...")
#     df = load_training_data_from_db(db_path)
#     with open(processed_file, "wb") as f:
#         pickle.dump(df, f)
#     print("Processed training data saved to file:", processed_file)
#     return df

# if __name__ == "__main__":
#     processed_file = r"C:\codes\t5-db\processed_training_data.pkl"
#     db_path = r"C:\codes\t5-db\researchers.db"
    
#     df = process_and_save_data(processed_file, db_path)
#     print(f"Retrieved {len(df)} training examples from the database.")
    
#     from llama_model import fine_tune_llama_on_papers
#     output_dir = fine_tune_llama_on_papers(df)
#     print("Fine-tuned LLaMA model saved at:", output_dir)

# train_llama.py
import sqlite3
import pandas as pd
import pickle

def load_metadata_training_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT researcher_name, work_title, authors, info FROM research_info"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Replace NaN
    df = df.fillna("")

    # Create training pairs
    rows = []
    for _, row in df.iterrows():
        author = row["researcher_name"]
        title = row["work_title"]
        authors = row["authors"]
        info = row["info"]

        prompt = f"""You are a Syracuse University research assistant.

Paper Title: {title}
Author(s): {authors}
Researcher Name: {author}
Publication Info: {info}

Summary:"""

        # Make target a narrative summary of metadata
        target = f"{title} is a research work authored by {authors} at Syracuse University. {info}".strip()

        rows.append({"input_text": prompt, "target_text": target})

    df_out = pd.DataFrame(rows)
    return df_out

def process_and_save_data(processed_file, db_path):
    df = load_metadata_training_data(db_path)
    with open(processed_file, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved metadata training data with {len(df)} samples to:", processed_file)
    return df

if __name__ == "__main__":
    processed_file = r"C:\codes\t5-db\processed_metadata_training.pkl"
    db_path = r"C:\codes\t5-db\researchers.db"

    df = process_and_save_data(processed_file, db_path)
    
    from llama_model import fine_tune_llama_on_papers
    output_dir = fine_tune_llama_on_papers(df)
    print("Fine-tuned LLaMA model saved at:", output_dir)
