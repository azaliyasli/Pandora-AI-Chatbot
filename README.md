# Pandora AI Chatbot

Pandora AI is an AI-powered emotional support chatbot designed to analyze user emotions and generate empathetic, context-aware responses. The system combines deep learning–based emotion detection with a Retrieval-Augmented Generation (RAG) approach to deliver meaningful and supportive conversations.

---

## 🚀 Project Overview

Pandora AI follows a **hybrid architecture** consisting of:

- **Emotion Classification** using a DistilBERT-based deep learning model
- **Retrieval-Augmented Generation (RAG)** for emotion-aware response selection
- **Google Gemini** is used as an LLM layer to generate richer and more empathetic responses 
- **User Authentication & Chat Persistence** via email-based login  
- **Web-based Chat Interface** for real-time interaction  

All user messages are stored in a database and are reloaded when the user logs in again with the same email.

---

## 🧠 Features

✔ Email-based user login  
✔ User-specific chat history persistence  
✔ Emotion detection with DistilBERT  
✔ RAG-based response generation
✔ LLM support using Google Gemini for generative responses
✔ Real-time chat interface
✔ Session handling & logout support  
✔ CPU-compatible and offline-friendly architecture  

---

## 🧩 Tech Stack

| Layer | Technology |
|------|-----------|
| NLP & Model Training | PyTorch, HuggingFace Transformers |
| Retrieval (RAG) | Sentence-Transformers, Cosine Similarity |
| LLM | Google Gemini API |
| Backend API | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Database | MySQL / External DB (configurable) |
| Deployment | Local (cloud deployment limited by memory constraints) |

---

## ⚙️ How It Works

1. The user logs in using an email address.
2. The user sends a message via the chat interface.
3. The backend analyzes the message with a DistilBERT emotion classifier.
4. Based on the detected emotion, the RAG module retrieves the most suitable response.
5. The response is enhanced using a Large Language Model (Google Gemini).
6. The response is sent back to the frontend.
7. The entire conversation is stored and restored on future logins.

---

## 🧪 Getting Started (Local)

### Clone the repository
```bash
git clone https://github.com/azaliyasli/Pandora-AI-Chatbot.git
cd Pandora-AI-Chatbot
