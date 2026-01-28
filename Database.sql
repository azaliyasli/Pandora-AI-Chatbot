CREATE DATABASE pandora_ai_db;
USE pandora_ai_db;

CREATE TABLE chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255),
    user_message TEXT,
    bot_response TEXT,
    emotion VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT *
FROM chat_history