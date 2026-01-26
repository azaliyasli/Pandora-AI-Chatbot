import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_pandora_response(user_text, emotion, rag_hint):
    model_id = "gemini-2.0-flash-lite"

    prompt = f"""
        Role: Pandora (Empathetic friend).
        User: "{user_text}"
        Emotion: {emotion}
        Strategy: {rag_hint}
        Task: Give a short, heartfelt response.
        """

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
if __name__ == "__main__":
    print(get_pandora_response("I'm so tired", "sad", "Respond with empathy."))