# modules/chatbot.py
from typing import List, Dict

def init_chat():
    # returns an empty list of messages
    # each message: {"role": "user"/"bot", "text": "..."}
    return []

def add_user_message(chat_history: List[Dict], text: str):
    chat_history.append({"role": "user", "text": text})
    return chat_history

def add_bot_message(chat_history: List[Dict], text: str):
    chat_history.append({"role": "bot", "text": text})
    return chat_history