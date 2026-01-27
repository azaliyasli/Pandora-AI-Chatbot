from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from rag import get_final_context
from llm import get_pandora_response
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/chat")
async def chat(data: ChatInput):
    print(f"DEBUG: Input -> {data.message}")
    try:
        emotion, hint = get_final_context(data.message)
        print(f"DEBUG: NLP Sentiment -> {emotion}")

        response = get_pandora_response(data.message, emotion, hint)
        print(f"DEBUG: Gemini Response -> {response}")

        return {
            "reply": response,
            "emotion": emotion
        }
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return {
            "reply": "I'm having a little trouble connecting right now, but I'm still listening. Please tell me more.",
            "emotion": "error"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)