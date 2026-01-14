'''Flask application that acts as the webhook endpoint for the Twilio WhatsApp chatbot.
It receives incoming WhatsApp messages, forwards them to the bot logic, and returns
the generated responses back to the user via Twilio.'''

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os

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
    - WhatsApp button/list replies
    """

    incoming_text = request.values.get("Body", "").strip()
    button_payload = request.values.get("ListReplyId") or request.values.get("ButtonPayload")

    # Priority: button payload > text
    user_input = button_payload if button_payload else incoming_text

    response_text, interactive = handle_message(user_input)

    resp = MessagingResponse()

    if interactive:
        # Send WhatsApp LIST MESSAGE
        msg = resp.message()
        msg.body(interactive["body"])

        msg.list(
            interactive["button"],
            interactive["sections"]
        )
    else:
        resp.message(response_text)

    return str(resp)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
