"""
Flask application that acts as the webhook endpoint for the Twilio WhatsApp chatbot.
It receives incoming WhatsApp messages, forwards them to the bot logic, and returns
the generated responses back to the user via Twilio.
"""

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

from bot import setup_bot, handle_message

load_dotenv()

app = Flask(__name__)
setup_bot()

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    """
    Main Twilio webhook
    Handles:
    - Text messages
    """

    incoming_text = request.values.get("Body", "").strip()

    response_text, _ = handle_message(incoming_text)

    resp = MessagingResponse()
    resp.message(response_text)

    return str(resp)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
