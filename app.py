import os
import mysql.connector
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from rag import get_final_context
from llm import get_pandora_response

load_dotenv()
app = FastAPI()

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
    email: str

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

@app.post("/chat")
async def chat(data: ChatInput):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT user_message, bot_response FROM chat_history WHERE email = %s ORDER BY created_at DESC LIMIT 5",
            (data.email,)
        )
        past_chats = cursor.fetchall()

        history_context = ""
        for chat_row in reversed(past_chats):
            history_context += f"User: {chat_row['user_message']}\nPandora: {chat_row['bot_response']}\n"

        emotion, hint = get_final_context(data.message)

        response = get_pandora_response(data.message, emotion, hint, history_context)

        cursor.execute(
            "INSERT INTO chat_history (email, user_message, bot_response, emotion) VALUES (%s, %s, %s, %s)",
            (data.email, data.message, response, emotion)
        )

        conn.commit()
        cursor.close()
        conn.close()

        return {
            "reply": response,
            "emotion": emotion
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return {
            "reply": "I'm having a little trouble remembering our history, but I'm here. How can I help?",
            "emotion": "error"
        }

@app.get("/history/{email}")
async def get_history(email: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT user_message, bot_response, emotion FROM chat_history WHERE email = %s ORDER BY created_at ASC",
            (email,)
        )
        history = cursor.fetchall()
        cursor.close()
        conn.close()
        return history # JSON listesi olarak döner
    except Exception as e:
        print(f"History Error: {e}")
        return []


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)