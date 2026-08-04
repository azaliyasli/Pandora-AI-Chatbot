import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_pandora_response(user_text, emotion, rag_hint, retrieved_context, history_context=""):
    model_id = "gemini-2.5-flash"

    prompt = f"""
        Role: Pandora (Empathetic AI friend).
        Past Conversations:
        {history_context}

        Current Emotion: {emotion}
        RAG Strategy: {rag_hint}
        Context (Similar past scenarios): 
        {retrieved_context}

        User's New Message: {user_text}

        Instruction: Use the past conversation context and the provided RAG Strategy to provide a more personal and caring response. Consider how similar scenarios were handled.
        Pandora:"""

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "I am here for you, always. Tell me more about how you feel."