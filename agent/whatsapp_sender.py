import os
import logging
from typing import Optional
from twilio.rest import Client

logger = logging.getLogger(__name__)

def send_whatsapp(message: str, to_number: Optional[str] = None) -> None:
    """Send a WhatsApp message using the Twilio WhatsApp Sandbox API.

    Args:
        message (str): The body text of the message to send.
        to_number (Optional[str]): Target phone number. If None, falls back to MY_WHATSAPP.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_whatsapp = os.environ.get("TWILIO_WHATSAPP_FROM")
    target_number = to_number or os.environ.get("MY_WHATSAPP")

    if not all([account_sid, auth_token, from_whatsapp, target_number]):
        logger.warning(
            "Twilio credentials (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM, "
            "MY_WHATSAPP) not fully set in environment. Skipping WhatsApp message dispatch."
        )
        return

    try:
        # Standardize the format with 'whatsapp:' prefix if not present
        if not from_whatsapp.startswith("whatsapp:"):
            from_whatsapp = f"whatsapp:{from_whatsapp}"
        if not target_number.startswith("whatsapp:"):
            target_number = f"whatsapp:{target_number}"

        client = Client(account_sid, auth_token)
        response = client.messages.create(
            from_=from_whatsapp,
            body=message,
            to=target_number
        )
        logger.info(f"WhatsApp notification sent successfully! SID: {response.sid}")
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message via Twilio: {str(e)}")
