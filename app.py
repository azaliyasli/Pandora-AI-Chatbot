import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from rag import get_final_context
from llm import get_pandora_response
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector

load_dotenv()
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=".")

class ChatInput(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

@app.post("/chat")
async def chat(data: ChatInput, email: str = "test@example.com"):
    try:

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT user_message, bot_response FROM chat_history WHERE email = %s ORDER BY created_at DESC LIMIT 5",
            (email,))
        past_chats = cursor.fetchall()

        history_context = ""
        for chat in reversed(past_chats):
            history_context += f"User: {chat['user_message']}\nPandora: {chat['bot_response']}\n"

        emotion, hint = get_final_context(data.message)

        response = get_pandora_response(data.message, emotion, hint, history_context)

        cursor.execute(
            "INSERT INTO chat_history (email, user_message, bot_response, emotion) VALUES (%s, %s, %s, %s)",
            (email, data.message, response, emotion)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"reply": response, "emotion": emotion}
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return {"reply": "I'm here, but I'm having trouble remembering our past. Let's talk anyway.",
                "emotion": "error"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)