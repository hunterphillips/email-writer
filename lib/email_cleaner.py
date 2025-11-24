"""
Email cleaning and extraction logic for Gmail .mbox files
"""

import mailbox
import re
from bs4 import BeautifulSoup
from lib.config import GENERIC_PROMPTS


def extract_clean_text(msg):
    """
    Extracts a clean body from an email message.
    Prioritizes text/plain, falls back to text/html.
    """
    text_part = None
    html_part = None

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = part.get("Content-Disposition", "")

            if "attachment" in disp:
                continue

            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            try:
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
            except:
                continue

            if ctype == "text/plain" and not text_part:
                text_part = decoded

            elif ctype == "text/html" and not html_part:
                soup = BeautifulSoup(decoded, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                html_part = soup.get_text("\n")

    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text_part = payload.decode(charset, errors="replace")

    body = text_part or html_part or ""

    # Final cleanup: remove any remaining HTML tags
    body = re.sub(r'<[^>]+>', '', body)
    # Clean up excessive whitespace
    body = re.sub(r'\n\s*\n\s*\n+', '\n\n', body)

    return body.strip()


def strip_quoted(body):
    """Remove quoted text from email replies."""
    # Remove quoted lines
    body = "\n".join([ln for ln in body.splitlines() if not ln.strip().startswith(">")])
    # Remove Gmail "On Tue, X wrote:" - more aggressive pattern
    body = re.split(r"On .+?wrote:", body, flags=re.DOTALL)[0]
    # Remove trailing quoted sections that start with "On [date]"
    body = re.split(r"\n\s*On\s+\w+,", body)[0]
    return body.strip()


def strip_signature(body):
    """Remove email signatures."""
    # Remove signature delimiter
    if "--" in body:
        body = body.split("--", 1)[0]
    # Remove "Sent from my iPhone"
    body = re.sub(r"Sent from my .+", "", body)
    return body.strip()


def strip_email_metadata(body):
    """Remove email headers and metadata that leak through."""
    # Remove Outlook separators
    body = re.sub(r'_{20,}', '', body)

    # Remove "From:", "Sent:", "To:", "Subject:" headers
    lines = body.splitlines()
    cleaned_lines = []

    for line in lines:
        # Skip header lines
        if re.match(r'^(From|Sent|To|Subject|Cc|Bcc):\s*', line, re.IGNORECASE):
            continue
        # Skip caution warnings
        if 'email originated from an external sender' in line.lower():
            continue
        if 'do not click links' in line.lower():
            continue
        # Skip warning symbols
        if line.strip().startswith('⚠'):
            continue
        cleaned_lines.append(line)

    body = '\n'.join(cleaned_lines)
    return body.strip()


# ==========================================
# Filtering functions
# ==========================================

def is_url_only(body):
    """Filter out URL-only emails or emails with just a URL and minimal context."""
    stripped = body.strip()
    # Pure URL
    url_pattern = r'^https?://\S+$'
    if re.match(url_pattern, stripped):
        return True

    # URL with minimal text (less than 10 words excluding the URL)
    text_without_urls = re.sub(r'https?://\S+', '', body)
    if len(text_without_urls.split()) < 10 and 'http' in body:
        return True

    return False


def is_auto_generated(body):
    """Filter auto-generated messages."""
    patterns = [
        "automatically generated",
        "do not reply",
        "this is an automated"
    ]
    return any(p in body.lower() for p in patterns)


def is_test_message(body, subject):
    """Filter test messages."""
    test_patterns = ["test forward", "test redirect", "test email"]
    combined = (body + " " + subject).lower()
    return any(p in combined for p in test_patterns)


def is_signature_only(body):
    """Filter signature-only emails (just name/phone/email)."""
    if len(body.split()) < 10:
        phone_pattern = r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        email_pattern = r'\S+@\S+'
        return bool(re.search(phone_pattern, body) or re.search(email_pattern, body))
    return False


def is_confirmation_email(body):
    """Filter confirmation/tracking emails."""
    patterns = [
        r'confirmation\s*#',
        r'order\s*#',
        r'tracking\s*number',
        r'flight\s*#'
    ]
    return any(re.search(p, body, re.IGNORECASE) for p in patterns)


def has_only_image_refs(body):
    """Filter emails with only image references."""
    if '[image:' in body.lower() or 'image.png' in body.lower():
        text_without_images = re.sub(r'\[image:.*?\]', '', body, flags=re.IGNORECASE)
        return len(text_without_images.split()) < 5
    return False


def has_code_or_css(body):
    """Filter emails containing CSS or JavaScript code."""
    code_patterns = [
        r'\{[\s\n]*[\w-]+\s*:\s*[\w-]+\s*;',  # CSS properties like { color: red; }
        r'#[\w_-]+\s*\{',  # CSS ID selectors
        r'\.[\w_-]+\s*\{',  # CSS class selectors
        r'function\s*\(',  # JavaScript function
        r'var\s+\w+\s*=',  # JavaScript var
        r'const\s+\w+\s*=',  # JavaScript const
    ]
    return any(re.search(pattern, body) for pattern in code_patterns)


def is_meeting_invite(body):
    """Filter meeting invite details (Zoom, Teams, etc.)."""
    meeting_patterns = [
        r'join\s+zoom\s+meeting',
        r'meeting\s+id:\s*\d+',
        r'passcode:\s*\d+',
        r'dial\s+by\s+your\s+location',
        r'join\s+teams\s+meeting',
        r'google\s+meet',
        r'one\s+tap\s+mobile',
    ]
    return any(re.search(pattern, body, re.IGNORECASE) for pattern in meeting_patterns)


def has_email_headers(body):
    """Filter emails with metadata headers leaked through."""
    header_patterns = [
        r'^From:\s+\S+',
        r'^Sent:\s+\w+,',
        r'^To:\s+\S+',
        r'^Subject:\s+.+',
        r'email originated from an external sender',
        r'________________________________',  # Outlook separator
    ]
    return any(re.search(pattern, body, re.MULTILINE | re.IGNORECASE) for pattern in header_patterns)


def has_form_data(body):
    """Filter emails with account/form data patterns."""
    form_patterns = [
        r'account:\s*\w+',
        r'username:\s*\w+',
        r'balance:\s*\$\d+',
        r'user\s+id:\s*\d+',
        r'password:\s*\w+',
    ]
    # Check if multiple form-like patterns exist
    matches = sum(1 for pattern in form_patterns if re.search(pattern, body, re.IGNORECASE))
    return matches >= 2  # If 2+ form fields, likely form data


def is_meaningful(body, subject=""):
    """
    Check if an email body is meaningful and worth including in training data.
    Returns True if the email should be kept, False if it should be filtered out.
    """
    if not body.strip():
        return False
    # Skip pure emoji reactions, e.g. ❤️ or 👍
    if re.fullmatch(r"[^\w\s]+", body.strip()):
        return False
    # Skip Gmail auto "reacted via Gmail"
    if "reacted via Gmail" in body:
        return False
    # Skip single-word "unsubscribe" messages
    if body.strip().lower() == "unsubscribe":
        return False
    # Skip URL-only emails
    if is_url_only(body):
        return False
    # Skip auto-generated messages
    if is_auto_generated(body):
        return False
    # Skip test messages
    if is_test_message(body, subject):
        return False
    # Skip signature-only emails
    if is_signature_only(body):
        return False
    # Skip confirmation emails
    if is_confirmation_email(body):
        return False
    # Skip image-only references
    if has_only_image_refs(body):
        return False
    # Skip code/CSS
    if has_code_or_css(body):
        return False
    # Skip meeting invites
    if is_meeting_invite(body):
        return False
    # Skip emails with metadata headers
    if has_email_headers(body):
        return False
    # Skip form data
    if has_form_data(body):
        return False
    return True


def intent_from_subject_or_body(subject, body):
    """Generate a synthetic intent prompt for non-reply emails."""
    s = subject.lower().strip()
    b = body.lower().strip()

    if "unsubscribe" in s or b == "unsubscribe":
        return "Write an email asking to unsubscribe."

    if s.startswith("re:"):
        return "Write an email responding to a message."

    if s.startswith("fwd:") or s.startswith("fw:"):
        return "Write a forwarded message."

    if len(body.split()) <= 4:
        return "Write a brief one-line email in your tone."

    return "Write an email in your tone."


def process_mbox(mbox_path, user_email):
    """
    Process an mbox file and extract sent emails into training examples.

    Args:
        mbox_path: Path to the .mbox file
        user_email: The user's email address (to identify sent messages)

    Returns:
        List of training examples in OpenAI format
    """
    box = mailbox.mbox(mbox_path)

    outbound = []
    inbound = {}

    for msg in box:
        msgid = msg.get("Message-ID", "").strip()
        from_addr = msg.get("From", "").lower()
        subject = msg.get("Subject", "")

        body = extract_clean_text(msg)
        body = strip_quoted(body)
        body = strip_signature(body)
        body = strip_email_metadata(body)

        if not is_meaningful(body, subject):
            continue

        # SENT by you
        if user_email.lower() in from_addr:
            outbound.append({
                "msgid": msgid,
                "inreply": msg.get("In-Reply-To", "").strip(),
                "subject": subject,
                "body": body
            })
        else:
            inbound[msgid] = body

    # Deduplicate outbound entries by message-id (Gmail stores duplicates)
    unique_outbound = {o["msgid"]: o for o in outbound}.values()

    dataset = []

    for msg in unique_outbound:
        if msg["inreply"] and msg["inreply"] in inbound:
            # Build reply example
            dataset.append({
                "messages": [
                    {"role": "user", "content": inbound[msg["inreply"]]},
                    {"role": "assistant", "content": msg["body"]}
                ]
            })
        else:
            # Outbound-only email → synthetic intent
            intent = intent_from_subject_or_body(msg["subject"], msg["body"])
            dataset.append({
                "messages": [
                    {"role": "user", "content": intent},
                    {"role": "assistant", "content": msg["body"]}
                ]
            })

    return dataset
