import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_pandora_response(user_text, emotion, rag_hint):
    model_id = "gemini-2.5-flash"

    prompt = f"Role: Pandora. User: {user_text}. Emotion: {emotion}. Hint: {rag_hint}. Emphatic response:"

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "I am here for you, always. Tell me more about how you feel."

# Test
#if __name__ == "__main__":
#    print("--- Pandora Emotional Support Bot ---")
#    print("Type 'exit' or 'quit' to stop the conversation.\n")
#
#    while True:
#        user_input = input("You: ")
#
#        if user_input.lower() in ["exit", "quit"]:
#            print("Pandora: I'm always here if you need me. Take care! <3")
#            break
#
#        try:
#            from rag import get_final_context
#
#            emotion, hint = get_final_context(user_input)
#
#            response = get_pandora_response(user_input, emotion, hint)
#
#            print(f"\n[Detected Emotion: {emotion}]")
#            print(f"Pandora: {response}\n")
#
#        except Exception as e:
#            print(f"An error occurred: {e}")