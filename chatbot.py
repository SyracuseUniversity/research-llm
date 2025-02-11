from llama_model import chatbot_answer

def run_chatbot():
    """
    Runs an interactive chatbot loop.
    - If the user's input is a greeting (e.g., "hello", "hi", "hey"), responds with a friendly message.
    - Otherwise, sends the question to chatbot_answer to generate a response.
    """
    print("Chatbot is ready! Type your research question (or 'exit' to quit):")
    greetings = {"hello", "hi", "hey"}
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        if question.lower() in greetings:
            answer = "Hello! How can I help you with your research questions today?"
        else:
            answer = chatbot_answer(question)
        print("Chatbot:", answer)

if __name__ == "__main__":
    run_chatbot()
