import os
import logging
from typing import Optional
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

logger = logging.getLogger(__name__)

def format_whatsapp_number(number_str: str) -> str:
    """Format a phone number into Twilio's WhatsApp E.164 format: whatsapp:+<country_code><number>.

    Args:
        number_str (str): The raw phone number string.

    Returns:
        str: Correctly formatted E.164 WhatsApp number string.
    """
    clean_number = number_str.strip()
    if clean_number.lower().startswith("whatsapp:"):
        clean_number = clean_number[9:].strip()
        
    # Remove any non-numeric characters except +
    digits_only = "".join(c for c in clean_number if c.isdigit())
    
    if clean_number.startswith("+"):
        return f"whatsapp:{clean_number}"
        
    if len(digits_only) == 10:
        return f"whatsapp:+91{digits_only}"
    elif len(digits_only) == 12 and digits_only.startswith("91"):
        return f"whatsapp:+{digits_only}"
    
    if digits_only:
        return f"whatsapp:+{digits_only}"
        
    return number_str

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

    # Validate all required environment variables
    missing = []
    if not account_sid: missing.append("TWILIO_ACCOUNT_SID")
    if not auth_token: missing.append("TWILIO_AUTH_TOKEN")
    if not from_whatsapp: missing.append("TWILIO_WHATSAPP_FROM")
    if not target_number: missing.append("MY_WHATSAPP")

    if missing:
        err_msg = f"Missing required environment variables for Twilio: {', '.join(missing)}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Format numbers to E.164 WhatsApp syntax
    from_whatsapp = format_whatsapp_number(from_whatsapp)
    target_number = format_whatsapp_number(target_number)

    # Truncate message to Twilio WhatsApp limit of 1600 characters
    if len(message) > 1600:
        message = message[:1597] + "..."
        logger.info("Message body exceeded 1600 characters; truncated to fit Twilio limits.")

    # Log variables before sending
    logger.info(f"Preparing to dispatch WhatsApp message via Twilio Sandbox:")
    logger.info(f"  Sender: {from_whatsapp}")
    logger.info(f"  Recipient: {target_number}")

    try:
        client = Client(account_sid, auth_token)
        response = client.messages.create(
            from_=from_whatsapp,
            body=message,
            to=target_number
        )
        
        # Log response details
        logger.info("Twilio API Response:")
        logger.info(f"  Message SID: {response.sid}")
        logger.info(f"  Status: {response.status}")
        logger.info(f"  Error Code: {response.error_code}")
        logger.info(f"  Error Message: {response.error_message}")
        
    except TwilioRestException as te:
        if te.code == 63015 or "63015" in str(te):
            err_msg = (
                f"Twilio Sandbox Opt-in Error (Code 63015): The recipient {target_number} has not joined "
                f"the Twilio WhatsApp Sandbox. Please instruct the user to send the sandbox keyword (e.g., "
                f"'join <sandbox-keyword>') to the sandbox number ({from_whatsapp}) first."
            )
            logger.error(err_msg)
            raise Exception(err_msg) from te
        else:
            logger.error(f"Failed to dispatch WhatsApp message via Twilio (Code {te.code}): {te.msg}")
            raise te
    except Exception as e:
        logger.error(f"Unexpected WhatsApp error: {str(e)}")
        raise e
