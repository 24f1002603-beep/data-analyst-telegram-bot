import os
import json
import logging
import threading
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO)

# 1. READ CONFIGURATION & LOG URL
# Set PUBLIC_LOG_URL in your .env file or host dashboard (e.g., https://your-app.onrender.com/run.jsonl)
PUBLIC_LOG_URL = os.getenv("PUBLIC_LOG_URL", "http://localhost:8000/run.jsonl")
LOG_FILE_PATH = "run.jsonl"

# Automatically create run.jsonl if it doesn't exist yet
if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "w") as f:
        pass

# 2. FASTAPI BACKGROUND SERVER (Serves run.jsonl to the grader's `wget`)
web_app = FastAPI()

@web_app.get("/run.jsonl")
def get_log():
    """Endpoint allowing the evaluation suite to download logs using wget."""
    return FileResponse(LOG_FILE_PATH, media_type="application/x-ndjson")

def start_web_server():
    """Starts the web server in a background thread."""
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(web_app, host="0.0.0.0", port=port)

# 3. OPENAI CLIENT INITIALIZATION (Via AIPipe Direct OpenAI Proxy)
client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

# Multi-turn conversation memory store: {chat_id: [messages]}
chat_histories = {}

def append_to_run_log(chat_id: int, user_query: str, raw_llm_response: str):
    """Automatically appends chat executions as JSON lines to run.jsonl."""
    log_entry = {
        "chat_id": chat_id,
        "input": user_query,
        "output": raw_llm_response
    }
    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# 4. TELEGRAM MESSAGE HANDLER
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user_text = update.message.text
    logging.info(f"Received question from Chat ID {chat_id}: {user_text}")

    # Enforce strict answer output structure
    system_instruction = (
        "You are an expert data analysis agent.\n"
        "1. Analyze the user's query carefully using accurate public/MOSPI data.\n"
        "2. Return ONLY a single valid JSON object containing EXACTLY two top-level keys:\n"
        "   - 'answer': MUST match the exact JSON structure/keys requested by the user prompt. DO NOT add extra keys, markdown formatting, or commentary.\n"
        f"   - 'log_url': Set strictly to this exact URL string: {PUBLIC_LOG_URL}\n"
        "3. Do NOT wrap your output in markdown code blocks like ```json ... ```."
    )

    # Initialize message history for new chat sessions
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [{"role": "system", "content": system_instruction}]

    # Append current user query
    chat_histories[chat_id].append({"role": "user", "content": user_text})

    try:
        # Request strict JSON object format from gpt-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_histories[chat_id],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw_output = response.choices[0].message.content.strip()

        # Parse output and explicitly guarantee correct log_url is attached
        parsed_json = json.loads(raw_output)
        parsed_json["log_url"] = PUBLIC_LOG_URL
        final_output = json.dumps(parsed_json)

        # Record assistant reply in multi-turn history & append to run.jsonl
        chat_histories[chat_id].append({"role": "assistant", "content": final_output})
        append_to_run_log(chat_id, user_text, final_output)

        # Send raw JSON string directly back to Telegram
        await update.message.reply_text(final_output)

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        error_payload = {
            "answer": {"error": f"Failed to process query: {str(e)}"},
            "log_url": PUBLIC_LOG_URL
        }
        await update.message.reply_text(json.dumps(error_payload))

# 5. BOT & SERVER LAUNCHER
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN missing in .env file!")

    # Start FastAPI server in a background thread BEFORE launching Telegram polling
    threading.Thread(target=start_web_server, daemon=True).start()
    print(f"Log web server running. Serving log file at: {PUBLIC_LOG_URL}")

    request_config = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
    app = ApplicationBuilder().token(token).request(request_config).build()

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    print("Telegram Bot is running with polling...")
    app.run_polling()