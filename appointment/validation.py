"""
Validation utilities for Leibniz Appointment FSM Microservice

Contains helper functions for data validation and formatting.

Reference:
    leibniz_agent/leibniz_appointment_fsm.py (lines 1178-1503) - Original validation functions
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

from leibniz_agent.services.appointment.models import (
    EMAIL_PATTERN, PHONE_PATTERN_INTL, PHONE_PATTERN_GERMAN, NAME_PATTERN,
    DATE_RELATIVE, DATE_WEEKDAY, DATE_FORMATTED, TIME_PATTERN, TIME_SIMPLE, TIME_RELATIVE
)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_name(name: str) -> Tuple[bool, str]:
    """
    Validate name format.

    Args:
        name: Full name string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not NAME_PATTERN.match(name):
        return False, "I didn't quite catch that."

    # Check for at least 2 parts (first and last name)
    parts = name.split()
    if len(parts) < 2:
        return False, "Could you give me your full name (first and last)?"

    # Reject single-character names
    if any(len(part) < 2 for part in parts):
        return False, "That name seems a bit short."

    # Reject common invalid inputs
    if name.lower() in ["test", "none", "n/a"]:
        return False, "I need your real name for the appointment."

    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email address format.

    Args:
        email: Email address string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not EMAIL_PATTERN.match(email):
        return False, "That doesn't look like a valid email address."

    # Check for common typos
    if email.endswith(".con"):
        return False, "Did you mean '.com' instead of '.con'?"

    return True, ""


def validate_phone(phone: str) -> Tuple[bool, str]:
    """
    Validate phone number format.

    Args:
        phone: Phone number string

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Remove formatting for validation
    digits_only = re.sub(r'\D', '', phone)

    # Check length
    if len(digits_only) < 10 or len(digits_only) > 15:
        return False, "That phone number doesn't look right."

    # Check for obviously invalid patterns
    if digits_only == "1234567890" or digits_only == "0000000000":
        return False, "That doesn't look like a real phone number."

    return True, ""


def normalize_phone(phone: str) -> Optional[str]:
    """
    Normalize phone number to standard format.

    Args:
        phone: Phone number string

    Returns:
        Normalized phone number or None if invalid
    """
    # Remove all non-digit characters except leading +
    if phone.startswith('+'):
        digits = '+' + re.sub(r'\D', '', phone[1:])
    else:
        digits = re.sub(r'\D', '', phone)

    # If starts with 0 (German format), convert to +49
    if digits.startswith('0'):
        digits = '+49' + digits[1:]

    # If no country code, assume German (+49)
    if not digits.startswith('+'):
        digits = '+49' + digits

    # Validate length: 10-15 digits (international standard)
    digit_count = len(re.sub(r'\D', '', digits))
    if digit_count < 10 or digit_count > 15:
        return None

    return digits


def format_phone_for_readback(phone: str) -> str:
    """
    Format phone number with spaces for clear readback.

    Args:
        phone: Phone number string

    Returns:
        Formatted phone number for TTS
    """
    if not phone:
        return ""

    # Remove all non-digit characters except +
    cleaned = re.sub(r'[^\d+]', '', phone)

    # If starts with +49 (Germany), format as: +49 XXX XXX XX XX
    if cleaned.startswith('+49'):
        digits = cleaned[3:]  # Remove +49
        # Group: area code (3), main (3-4), rest (2-2)
        if len(digits) >= 9:
            formatted = f"+49 {digits[:3]} {digits[3:6]} {digits[6:8]}"
            if len(digits) > 8:
                formatted += f" {digits[8:]}"
            return formatted.strip()

    # If starts with +1 (US/Canada), format as: +1 XXX XXX XXXX
    elif cleaned.startswith('+1'):
        digits = cleaned[2:]
        if len(digits) == 10:
            return f"+1 {digits[:3]} {digits[3:6]} {digits[6:]}"

    # Generic formatting: +XX XXX XXX XXX
    if cleaned.startswith('+'):
        country_code = cleaned[:3]
        rest = cleaned[3:]
        # Group remaining digits in chunks of 3
        chunks = [rest[i:i+3] for i in range(0, len(rest), 3)]
        return f"{country_code} {' '.join(chunks)}"

    # Fallback: return as-is with spaces every 3 digits
    chunks = [cleaned[i:i+3] for i in range(0, len(cleaned), 3)]
    return ' '.join(chunks)


def spell_out_name(name: str) -> str:
    """
    Spell out name letter-by-letter for confirmation.

    Args:
        name: Name string

    Returns:
        Spelled out name (e.g., 'J-O-H-N S-M-I-T-H')
    """
    if not name:
        return ""

    words = name.split()
    spelled_words = []

    for word in words:
        # Spell each letter separated by hyphens
        letters = []
        for char in word:
            if char.isalpha():
                letters.append(char.upper())
            elif char == "'":
                letters.append("apostrophe")
            elif char == "-":
                letters.append("hyphen")

        spelled_word = "-".join(letters)
        spelled_words.append(spelled_word)

    # Join words with space
    return " ".join(spelled_words)


def parse_yes_no(user_input: str) -> Optional[bool]:
    """
    Parse yes/no response from user input.

    Args:
        user_input: User input string

    Returns:
        True for yes, False for no, None for unclear
    """
    if not user_input:
        return None

    cleaned = user_input.strip().lower()

    # Yes indicators
    yes_words = ["yes", "yeah", "yep", "correct", "right", "that's right",
                 "yup", "uh huh", "mhmm", "sure", "okay", "ok", "affirmative"]
    if any(word in cleaned for word in yes_words):
        return True

    # No indicators
    no_words = ["no", "nope", "wrong", "incorrect", "not right", "nah",
                "negative", "false", "that's wrong"]
    if any(word in cleaned for word in no_words):
        return False

    # Unclear/ambiguous
    return None


def parse_datetime(user_input: str) -> Optional[str]:
    """
    Parse date and time from natural language text.

    Args:
        user_input: Natural language datetime string

    Returns:
        Formatted datetime string or None if unparseable
    """
    text_lower = user_input.lower()
    now = datetime.now()
    parsed_date = None
    parsed_time = None

    # Parse natural language dates
    if "today" in text_lower:
        parsed_date = now
    elif "tomorrow" in text_lower:
        parsed_date = now + timedelta(days=1)
    elif "next week" in text_lower:
        parsed_date = now + timedelta(days=7)
    elif "next month" in text_lower:
        # Approximate next month as 30 days
        parsed_date = now + timedelta(days=30)

    # Parse weekdays
    weekday_match = DATE_WEEKDAY.search(text_lower)
    if weekday_match:
        target_weekday = weekday_match.group(1)
        weekday_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        target_num = weekday_map[target_weekday]
        current_weekday = now.weekday()
        days_ahead = (target_num - current_weekday) % 7
        if days_ahead == 0:
            days_ahead = 7  # Next occurrence
        parsed_date = now + timedelta(days=days_ahead)

    # Parse formatted dates (numeric formats)
    date_match = DATE_FORMATTED.search(user_input)
    if date_match:
        date_str = date_match.group(0)
        # Try different formats
        for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%m/%d/%y", "%d/%m/%y"]:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

    # Parse textual month formats
    if not parsed_date:
        # Try full month name with year: "December 25 2024"
        for fmt in ["%B %d %Y", "%b %d %Y"]:
            try:
                parsed_date = datetime.strptime(user_input, fmt)
                break
            except ValueError:
                continue

        # Try month name without year (default to current year)
        if not parsed_date:
            for fmt in ["%B %d", "%b %d"]:
                try:
                    parsed_date = datetime.strptime(user_input, fmt)
                    # Set year to current year
                    parsed_date = parsed_date.replace(year=now.year)
                    break
                except ValueError:
                    continue

    # Parse time formats
    time_match = TIME_PATTERN.search(text_lower)
    if time_match:
        time_str = time_match.group(0)
        # Try parsing with AM/PM
        for fmt in ["%I:%M %p", "%I:%M%p", "%H:%M"]:
            try:
                time_obj = datetime.strptime(time_str.upper(), fmt)
                parsed_time = time_obj.time()
                break
            except ValueError:
                continue

    # Parse simple time (e.g., "2pm")
    if not parsed_time:
        simple_time_match = TIME_SIMPLE.search(text_lower)
        if simple_time_match:
            time_str = simple_time_match.group(0)
            try:
                time_obj = datetime.strptime(time_str.upper(), "%I%p")
                parsed_time = time_obj.time()
            except ValueError:
                try:
                    time_obj = datetime.strptime(time_str.upper(), "%I %p")
                    parsed_time = time_obj.time()
                except ValueError:
                    pass

    # Parse relative time (morning, afternoon, evening)
    if not parsed_time:
        relative_time_match = TIME_RELATIVE.search(text_lower)
        if relative_time_match:
            time_word = relative_time_match.group(1)
            if time_word == "morning":
                parsed_time = datetime.strptime("10:00", "%H:%M").time()
            elif time_word == "afternoon":
                parsed_time = datetime.strptime("14:00", "%H:%M").time()
            elif time_word == "evening":
                parsed_time = datetime.strptime("17:00", "%H:%M").time()

    # Combine date and time
    if parsed_date and parsed_time:
        combined = datetime.combine(parsed_date.date(), parsed_time)
        return combined.strftime("%A, %B %d, %Y at %I:%M %p")
    elif parsed_date:
        # Default time if only date provided
        default_time = datetime.strptime("10:00", "%H:%M").time()
        combined = datetime.combine(parsed_date.date(), default_time)
        return combined.strftime("%A, %B %d, %Y at %I:%M %p")
    elif parsed_time:
        # Use tomorrow if only time provided
        tomorrow = now + timedelta(days=1)
        combined = datetime.combine(tomorrow.date(), parsed_time)
        return combined.strftime("%A, %B %d, %Y at %I:%M %p")

    return None


def validate_datetime(datetime_str: str) -> Tuple[bool, str]:
    """
    Validate parsed datetime against future date, booking window, and business hours.

    Args:
        datetime_str: Formatted datetime string

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Parse the formatted datetime string back to datetime object
    try:
        dt = datetime.strptime(datetime_str, "%A, %B %d, %Y at %I:%M %p")
    except ValueError:
        return False, "Could not parse the datetime format."

    now = datetime.now()

    # 1. Check if date is in the future
    if dt <= now:
        return False, "That time has already passed. Please choose a future date and time."

    # 2. Check maximum booking window (3 months default)
    max_future_date = now + timedelta(days=90)
    if dt > max_future_date:
        return False, "We can only book appointments up to 3 months in advance. Please choose a closer date."

    # 3. Check business hours (Mon-Fri 8am-6pm default)
    appointment_time = dt.time()
    start_time = datetime.strptime("08:00", "%H:%M").time()
    end_time = datetime.strptime("18:00", "%H:%M").time()

    if appointment_time < start_time or appointment_time >= end_time:
        return False, "That time is outside our business hours (8am-6pm). Please choose a time during business hours."

    return True, ""