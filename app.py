from flask import Flask, request
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
from bot import setup_bot, search_answer

load_dotenv()

app = Flask(__name__)

setup_bot()  # load CSV once


@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    user_msg = request.values.get("Body", "").strip()
    print("📩 Incoming:", user_msg)

    answer = search_answer(user_msg)

    resp = MessagingResponse()
    resp.message(answer)
    return str(resp)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
