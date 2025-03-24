"""
chatbot.py

This module implements an interactive chatbot that acts solely as a Syracuse subject matter expert.
When a user asks a question, the chatbot queries the 'research_info' table for records
that match the query (searching in researcher_name, work_title, or authors).
If matches are found, it returns the formatted record(s); otherwise, it replies with "Not found."
"""

import sqlite3

def get_research_info(query, db_path=r"C:\codes\t5-db\researchers.db"):
    """
    Query the 'research_info' table for records where the researcher_name, work_title, or authors 
    contain the query substring (case-insensitive). Returns a formatted answer if found, otherwise None.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        sql = """
            SELECT researcher_name, work_title, authors, info 
            FROM research_info 
            WHERE LOWER(researcher_name) LIKE ? 
               OR LOWER(work_title) LIKE ? 
               OR LOWER(authors) LIKE ?
        """
        pattern = "%" + query.lower() + "%"
        cursor.execute(sql, (pattern, pattern, pattern))
        results = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Error fetching research info: {e}")
        return None

    if results:
        answer_lines = []
        for rec in results:
            researcher_name, work_title, authors, info = rec
            answer_lines.append(
                f"Researcher: {researcher_name}\nWork Title: {work_title}\nAuthors: {authors}\nInfo: {info}"
            )
        return "\n\n".join(answer_lines)
    else:
        return None

def run_chatbot():
    """
    Runs an interactive chatbot loop.
    - Responds to greetings with a friendly message.
    - For other queries, it searches the 'research_info' table.
      If a match is found, returns the formatted record(s); otherwise, replies "Not found."
    """
    print("Chatbot is ready! Type your research question (or 'exit' to quit):")
    greetings = {"hello", "hi", "hey"}
    
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        if query.lower() in greetings:
            answer = "Hello! How can I help you with your research questions today?"
        else:
            info = get_research_info(query)
            if info:
                answer = info
            else:
                answer = "Not found."
        print("Chatbot:", answer)

if __name__ == "__main__":
    run_chatbot()
