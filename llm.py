import google.generativeai as genai
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# JSON File
with open('intents.json', 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

# Define instructions
system_instruction = f"""
Senin adın Pandora. Duygusal destek veren bir AI asistansısın. 
Aşağıdaki kurallara ve bilgilere göre konuşmalısın:
- Kullanıcı üzgünse empati yap.
- Eğer kullanıcı intihardan bahsederse mutlaka yardım hattını (9152987821) paylaş.
- Yanıtlarını şu bilgi setine dayandır: {json.dumps(intents_data['intents'][:10])}... (Verinin bir kısmını örnek olarak veriyoruz)
"""

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=system_instruction
)

# Chat
chat = model.start_chat(history=[])

while True:
    user_input = input("Sen: ")
    if user_input.lower() in ["exit", "kapat"]:
        break

    response = chat.send_message(user_input)
    print(f"Pandora: {response.text}")