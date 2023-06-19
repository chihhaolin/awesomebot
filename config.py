
with open("key.txt") as f:
    ai_key = f.read().strip()

OPENAI_KEY = ai_key

MEMORY_KEY = "chat_history"