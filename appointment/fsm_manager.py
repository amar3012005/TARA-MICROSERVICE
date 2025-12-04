"""
FSM Manager for Leibniz Appointment FSM Microservice

Contains the core finite state machine logic for appointment booking.

Reference:
    leibniz_agent/leibniz_appointment_fsm.py (lines 180-1067) - Original FSM logic
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from leibniz_agent.services.appointment.config import AppointmentConfig
from leibniz_agent.services.appointment.models import AppointmentData, AppointmentState, DEPARTMENTS, APPOINTMENT_TYPES, CANCEL_KEYWORDS
from leibniz_agent.services.appointment.validation import (
    validate_name, validate_email, validate_phone, normalize_phone,
    format_phone_for_readback, spell_out_name, parse_yes_no,
    parse_datetime, validate_datetime
)

logger = logging.getLogger(__name__)


class AppointmentFSMManager:
    """
    Finite State Machine for Leibniz University appointment booking.

    Manages the conversation flow for collecting appointment information
    without Redis persistence (stateless per-request).
    """

    def __init__(self, config: AppointmentConfig):
        """
        Initialize the appointment booking FSM manager.

        Args:
            config: Appointment configuration
        """
        self.config = config

        # State management
        self.state = AppointmentState.INIT
        self.data = AppointmentData()

        # Retry tracking (per field)
        self.retry_counts: Dict[str, int] = {}

        # Confirmation attempt tracking (per field)
        self.confirmation_attempts: Dict[str, int] = {}

        # Error tracking
        self.last_error: Optional[str] = None

        # Conversation history for context
        self.conversation_history: list[str] = []

        if self.config.log_state_transitions:
            logger.info(" AppointmentFSMManager initialized")

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input based on current FSM state.

        Args:
            user_input: User's spoken/typed input

        Returns:
            Dictionary with response, state, and metadata
        """
        # Store previous state for progression tracking
        previous_state = self.state

        # Store input in conversation history
        self.conversation_history.append(user_input)

        # Check for cancellation
        if any(keyword in user_input.lower() for keyword in CANCEL_KEYWORDS):
            self.state = AppointmentState.CANCELLED
            return {
                "response": "No problem! If you'd like to book an appointment later, just let me know. Is there anything else I can help you with?",
                "state": self.state.value,
                "previous_state": previous_state.value,
                "complete": False,
                "cancelled": True,
                "data": None,
                "error": None,
                "success": True
            }

        # Route to appropriate handler based on current state
        try:
            if self.state == AppointmentState.INIT:
                response = self._handle_init()
            elif self.state == AppointmentState.COLLECT_NAME:
                response = self._handle_name_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_NAME:
                response = self._handle_name_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_EMAIL:
                response = self._handle_email_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_EMAIL:
                response = self._handle_email_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_PHONE:
                response = self._handle_phone_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_PHONE:
                response = self._handle_phone_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_DEPARTMENT:
                response = self._handle_department_selection(user_input)
            elif self.state == AppointmentState.CONFIRM_DEPARTMENT:
                response = self._handle_department_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_APPOINTMENT_TYPE:
                response = self._handle_appointment_type_selection(user_input)
            elif self.state == AppointmentState.CONFIRM_APPOINTMENT_TYPE:
                response = self._handle_appointment_type_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_DATETIME:
                response = self._handle_datetime_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_DATETIME:
                response = self._handle_datetime_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_PURPOSE:
                response = self._handle_purpose_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_PURPOSE:
                response = self._handle_purpose_confirmation(user_input)
            elif self.state == AppointmentState.CONFIRM:
                response = self._handle_confirmation(user_input)
            elif self.state == AppointmentState.COMPLETE:
                response = "Your appointment is already booked! Is there anything else I can help you with?"
            elif self.state == AppointmentState.CANCELLED:
                response = "The appointment booking was cancelled. Would you like to start a new booking?"
            else:
                response = "I'm not sure what happened. Let's start over with the appointment booking."
                self.state = AppointmentState.INIT

            # Return response dictionary
            return {
                "response": response,
                "state": self.state.value,
                "previous_state": previous_state.value,
                "complete": self.state == AppointmentState.COMPLETE,
                "cancelled": self.state == AppointmentState.CANCELLED,
                "data": self.data.to_dict() if self.state == AppointmentState.COMPLETE else None,
                "error": self.last_error,
                "success": self.last_error is None  # Add success field
            }

        except Exception as e:
            logger.error(f" Error processing input in state {self.state}: {e}")
            return {
                "response": "Oops, something went wrong. Let me try that again. Could you repeat what you just said?",
                "state": self.state.value,
                "previous_state": previous_state.value,
                "complete": False,
                "cancelled": False,
                "data": None,
                "error": str(e),
                "success": False
            }

    # ========================================================================
    # State Handler Methods
    # ========================================================================

    def _handle_init(self) -> str:
        """Initialize appointment booking process"""
        response = (
            "Great! I'd be happy to help you schedule an appointment. "
            "I'll need to collect a few details from you—this should only take a couple of minutes.\n\n"
            "I'll ask for your name, contact info, which department you'd like to meet with, "
            "and what the appointment's for. Feel free to say 'cancel' at any time if you change your mind.\n\n"
            "Let's start with your full name. What's your name?"
        )

        # Transition to name collection
        self.state = AppointmentState.COLLECT_NAME

        return response

    def _handle_name_collection(self, user_input: str) -> str:
        """Collect and validate user's full name"""
        # Clean input
        cleaned = user_input.strip()

        # Remove common prefixes
        prefixes = ["my name is", "i'm", "this is", "it's", "name:", "i am"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # Extract name
        name = cleaned.strip()

        # Validate name
        is_valid, error_msg = validate_name(name)

        if is_valid:
            # Success path
            self.data.name = name
            self.retry_counts['name'] = 0
            self.state = AppointmentState.CONFIRM_NAME

            # Transition to name confirmation
            spelled_name = spell_out_name(name)
            return f"Got it! So your name is {spelled_name}, am I right?"
        else:
            # Failure path
            self.retry_counts['name'] = self.retry_counts.get('name', 0) + 1
            self.last_error = error_msg

            if self.retry_counts['name'] < self.config.max_retries:
                return f"{error_msg} Could you tell me your full name? For example, 'John Smith' or 'Maria Garcia'. Take your time!"
            else:
                return "I'm having trouble getting your name. Would you like to try again, or should we cancel the appointment booking?"

    def _handle_name_confirmation(self, user_input: str) -> str:
        """Handle confirmation of collected name"""
        response_type = parse_yes_no(user_input)

        if response_type is True:
            # Confirmed - proceed to email collection
            self.confirmation_attempts['name'] = 0
            self.state = AppointmentState.COLLECT_EMAIL
            return f"Awesome— nice to meet you, {self.data.name.split()[0]}! Now, what's your email address? I'll send the appointment confirmation there."

        elif response_type is False:
            # Rejected - go back to name collection
            self.data.name = None
            self.confirmation_attempts['name'] = 0
            self.state = AppointmentState.COLLECT_NAME
            return "No worries! What's your name? Take your time!"

        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['name'] = self.confirmation_attempts.get('name', 0) + 1

            if self.confirmation_attempts['name'] >= self.config.max_confirmation_attempts:
                # Default to yes after max empty attempts
                if self.config.log_state_transitions:
                    logger.info(f"Name confirmation: defaulting to 'yes' after {self.confirmation_attempts['name']} empty responses")
                self.confirmation_attempts['name'] = 0
                self.state = AppointmentState.COLLECT_EMAIL
                return f"Perfect! Now, what's your email address? I'll send the appointment confirmation there."
            else:
                # Repeat confirmation
                spelled_name = spell_out_name(self.data.name)
                return f"So your name is {spelled_name}, am I right? Please say 'yes' or 'no'."

    def _handle_email_collection(self, user_input: str) -> str:
        """Collect and validate email address"""
        # Clean input
        cleaned = user_input.strip().lower()

        # Remove common phrases
        prefixes = ["my email is", "it's", "email:", "it is", "the email is"]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # Extract email
        email = cleaned.strip()

        # Validate email
        is_valid, error_msg = validate_email(email)

        if is_valid:
            # Success path
            self.data.email = email
            self.retry_counts['email'] = 0
            self.state = AppointmentState.CONFIRM_EMAIL

            # Transition to email confirmation
            return f"Great! So your email is {email}, am I right?"
        else:
            # Invalid email format
            self.retry_counts['email'] = self.retry_counts.get('email', 0) + 1
            self.last_error = error_msg

            if self.retry_counts['email'] < self.config.max_retries:
                return f"{error_msg} Could you try again? For example, 'john.smith@uni-hannover.de'. No worries if you need a moment!"
            else:
                # Default to placeholder and proceed
                self.data.email = "Not provided"
                self.retry_counts['email'] = 0
                self.state = AppointmentState.CONFIRM_EMAIL
                return f"No problem, we'll continue without the email for now. So we'll use '{self.data.email}' as your email, am I right?"

    def _handle_email_confirmation(self, user_input: str) -> str:
        """Handle confirmation of collected email"""
        response_type = parse_yes_no(user_input)

        if response_type is True:
            # Confirmed - proceed to phone collection
            self.confirmation_attempts['email'] = 0
            self.state = AppointmentState.COLLECT_PHONE
            return "Perfect! And what's your phone number? Include the country code if you're calling from outside Germany."

        elif response_type is False:
            # Rejected - go back to email collection
            self.data.email = None
            self.confirmation_attempts['email'] = 0
            self.state = AppointmentState.COLLECT_EMAIL
            return "No problem! What's your email address? Feel free to spell it out!"

        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['email'] = self.confirmation_attempts.get('email', 0) + 1

            if self.confirmation_attempts['email'] >= self.config.max_confirmation_attempts:
                # Default to yes after max empty attempts
                if self.config.log_state_transitions:
                    logger.info(f"Email confirmation: defaulting to 'yes' after {self.confirmation_attempts['email']} empty responses")
                self.confirmation_attempts['email'] = 0
                self.state = AppointmentState.COLLECT_PHONE
                return "Perfect! And what's your phone number? Include the country code if you're calling from outside Germany."
            else:
                # Repeat confirmation
                return f"So your email is {self.data.email}, am I right? Please say 'yes' or 'no'."

    def _handle_phone_collection(self, user_input: str) -> str:
        """Collect and validate phone number"""
        # Clean input
        cleaned = user_input.strip()

        # Remove common phrases
        prefixes = ["my number is", "phone:", "call me at", "it's", "the number is"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # Extract phone
        phone = cleaned.strip()

        # Normalize phone
        normalized = normalize_phone(phone)

        if normalized:
            # Validate phone
            is_valid, error_msg = validate_phone(normalized)

            if is_valid:
                # Success path
                self.data.phone = normalized
                self.retry_counts['phone'] = 0
                self.state = AppointmentState.CONFIRM_PHONE

                # Transition to phone confirmation
                formatted_phone = format_phone_for_readback(normalized)
                return f"Perfect! So your phone number is {formatted_phone}, am I right?"
            else:
                # Invalid phone
                self.retry_counts['phone'] = self.retry_counts.get('phone', 0) + 1
                self.last_error = error_msg

                if self.retry_counts['phone'] < self.config.max_retries:
                    return f"{error_msg} Could you try again? For example, '+49 511 762 2020' or '0511 762 2020'. No worries if it's tricky!"
                else:
                    # Default to placeholder and proceed
                    self.data.phone = "Not provided"
                    self.retry_counts['phone'] = 0
                    self.state = AppointmentState.CONFIRM_PHONE
                    return f"That's okay, let's continue. So we'll use '{self.data.phone}' as your phone number, am I right?"
        else:
            # Normalization failed
            self.retry_counts['phone'] = self.retry_counts.get('phone', 0) + 1

            if self.retry_counts['phone'] < self.config.max_retries:
                return "I couldn't understand that phone number. Could you try again? For example, '+49 511 762 2020'. Take your time!"
            else:
                # Default to placeholder and proceed
                self.data.phone = "Not provided"
                self.retry_counts['phone'] = 0
                self.state = AppointmentState.CONFIRM_PHONE
                return f"No problem, let's move on. So we'll use '{self.data.phone}' as your phone number, am I right?"

    def _handle_phone_confirmation(self, user_input: str) -> str:
        """Handle confirmation of collected phone number"""
        response_type = parse_yes_no(user_input)

        if response_type is True:
            # Confirmed - proceed to department selection
            self.confirmation_attempts['phone'] = 0
            self.state = AppointmentState.COLLECT_DEPARTMENT

            # Generate department options list
            dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
            return f"Great! Now, which department would you like to schedule an appointment with? Here are your options:\n\n{dept_list}\n\nFeel free to say the number or the department name."

        elif response_type is False:
            # Rejected - go back to phone collection
            self.data.phone = None
            self.confirmation_attempts['phone'] = 0
            self.state = AppointmentState.COLLECT_PHONE
            return "No worries! What's your phone number? Feel free to spell it out if needed!"

        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['phone'] = self.confirmation_attempts.get('phone', 0) + 1

            if self.confirmation_attempts['phone'] >= self.config.max_confirmation_attempts:
                # Default to yes after max empty attempts
                if self.config.log_state_transitions:
                    logger.info(f"Phone confirmation: defaulting to 'yes' after {self.confirmation_attempts['phone']} empty responses")
                self.confirmation_attempts['phone'] = 0
                self.state = AppointmentState.COLLECT_DEPARTMENT
                dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
                return f"Perfect! Which department would you like to schedule an appointment with?\n\n{dept_list}\n\nFeel free to say the number or the department name."
            else:
                # Repeat confirmation
                formatted_phone = format_phone_for_readback(self.data.phone)
                return f"So your phone number is {formatted_phone}, am I right? Please say 'yes' or 'no'."

    def _handle_department_selection(self, user_input: str) -> str:
        """Select department/service from available options"""
        # Clean input
        cleaned = user_input.strip().lower()

        # Try numbered selection first
        if cleaned.isdigit():
            index = int(cleaned) - 1
            dept_keys = list(DEPARTMENTS.keys())
            if 0 <= index < len(dept_keys):
                # Valid number selection
                self.data.department = dept_keys[index]
                self.retry_counts['department'] = 0
                self.state = AppointmentState.CONFIRM_DEPARTMENT

                # Transition to department confirmation
                return f"Got it! So you want to meet with {DEPARTMENTS[self.data.department]}, am I right?"

        # Match against department keywords
        matched_dept = None

        # Department keyword matching
        if "academic" in cleaned or "advising" in cleaned or "advisor" in cleaned:
            matched_dept = "academic_advising"
        elif "faculty" in cleaned or "professor" in cleaned:
            matched_dept = "faculty"
        elif "exam" in cleaned or "examination" in cleaned or "grade" in cleaned:
            matched_dept = "examination_office"
        elif "international" in cleaned or "visa" in cleaned or "foreign" in cleaned:
            matched_dept = "international_office"
        elif "career" in cleaned or "job" in cleaned:
            matched_dept = "career_services"
        elif "counseling" in cleaned or "psychological" in cleaned or "mental" in cleaned or "therapy" in cleaned:
            matched_dept = "counseling"
        elif "financial" in cleaned or "scholarship" in cleaned or "aid" in cleaned or "money" in cleaned:
            matched_dept = "financial_aid"
        elif "admission" in cleaned or "apply" in cleaned or "application" in cleaned:
            matched_dept = "academic_advising"  # Map admissions to academic advising
        elif "registration" in cleaned or "enrollment" in cleaned or "register" in cleaned:
            matched_dept = "registration"
        # IT matching with word boundaries
        elif (re.search(r'\bit\b', cleaned) or "IT support" in user_input or "IT services" in user_input or
              "technical help" in cleaned or "computer account" in cleaned or "tech support" in cleaned):
            matched_dept = "it_services"

        if matched_dept:
            # Success path
            self.data.department = matched_dept
            self.retry_counts['department'] = 0
            self.state = AppointmentState.CONFIRM_DEPARTMENT

            # Transition to department confirmation
            return f"Perfect! So you want to meet with {DEPARTMENTS[matched_dept]}, am I right?"
        else:
            # Failure path
            self.retry_counts['department'] = self.retry_counts.get('department', 0) + 1

            if self.retry_counts['department'] < self.config.max_retries:
                dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
                return f"I didn't catch which department you need. Could you choose from this list?\n\n{dept_list}\n\nFeel free to say the number or the department name. Take your time!"
            else:
                # Default to academic advising
                self.data.department = "academic_advising"
                self.retry_counts['department'] = 0
                self.state = AppointmentState.CONFIRM_DEPARTMENT
                return f"Let me put you down for {DEPARTMENTS[self.data.department]}—we can change this later if needed. So you want to meet with {DEPARTMENTS[self.data.department]}, am I right?"

    def _handle_department_confirmation(self, user_input: str) -> str:
        """Handle confirmation of selected department"""
        response_type = parse_yes_no(user_input)

        if response_type is True:
            # Confirmed - proceed to appointment type selection
            self.confirmation_attempts['department'] = 0
            self.state = AppointmentState.COLLECT_APPOINTMENT_TYPE

            # Get appointment types for selected department
            types = APPOINTMENT_TYPES[self.data.department]
            types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(types)])
            return f"Perfect! For {DEPARTMENTS[self.data.department]}, what type of appointment do you need?\n\n{types_list}\n\nYou can say the number or the appointment type."

        elif response_type is False:
            # Rejected - go back to department selection
            self.data.department = None
            self.confirmation_attempts['department'] = 0
            self.state = AppointmentState.COLLECT_DEPARTMENT

            dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
            return f"No problem! Which department would you like?\n\n{dept_list}\n\nFeel free to say the number or the department name."

        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['department'] = self.confirmation_attempts.get('department', 0) + 1

            if self.confirmation_attempts['department'] >= self.config.max_confirmation_attempts:
                # Default to yes after max empty attempts
                if self.config.log_state_transitions:
                    logger.info(f"Department confirmation: defaulting to 'yes' after {self.confirmation_attempts['department']} empty responses")
                self.confirmation_attempts['department'] = 0
                self.state = AppointmentState.COLLECT_APPOINTMENT_TYPE
                types = APPOINTMENT_TYPES[self.data.department]
                types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(types)])
                return f"Perfect! For {DEPARTMENTS[self.data.department]}, what type of appointment do you need?\n\n{types_list}\n\nYou can say the number or the appointment type."
            else:
                # Repeat confirmation
                return f"So you want to meet with {DEPARTMENTS[self.data.department]}, am I right? Please say 'yes' or 'no'."

    def _handle_appointment_type_selection(self, user_input: str) -> str:
        """Select appointment type based on chosen department"""
        # Get available types for selected department
        available_types = APPOINTMENT_TYPES[self.data.department]

        # Clean input
        cleaned = user_input.strip().lower()

        # Try numbered selection first
        if cleaned.isdigit():
            index = int(cleaned) - 1
            if 0 <= index < len(available_types):
                # Valid number selection
                self.data.appointment_type = available_types[index]
                self.retry_counts['appointment_type'] = 0
                self.state = AppointmentState.CONFIRM_APPOINTMENT_TYPE

                # Transition to appointment type confirmation
                return f"Excellent! So you need {self.data.appointment_type}, am I right?"

        # Match against appointment type keywords
        matched_type = None

        for atype in available_types:
            # Check if any word from the type appears in user input
            type_words = atype.lower().split()
            if any(word in cleaned for word in type_words):
                matched_type = atype
                break

        if matched_type:
            # Success path
            self.data.appointment_type = matched_type
            self.retry_counts['appointment_type'] = 0
            self.state = AppointmentState.CONFIRM_APPOINTMENT_TYPE

            # Transition to appointment type confirmation
            return f"Sounds good! So you need {matched_type}, am I right?"
        else:
            # Failure path
            self.retry_counts['appointment_type'] = self.retry_counts.get('appointment_type', 0) + 1

            if self.retry_counts['appointment_type'] < self.config.max_retries:
                types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(available_types)])
                return f"I'm not sure which type you need. Here are the options again:\n\n{types_list}\n\nFeel free to say the number or the appointment type. No worries if you need a moment!"
            else:
                # Default to first type
                self.data.appointment_type = available_types[0]
                self.retry_counts['appointment_type'] = 0
                self.state = AppointmentState.CONFIRM_APPOINTMENT_TYPE
                return f"I'll put you down for {available_types[0]}—we can adjust this later. So you need {self.data.appointment_type}, am I right?"

    def _handle_appointment_type_confirmation(self, user_input: str) -> str:
        """Handle confirmation of selected appointment type"""
        response_type = parse_yes_no(user_input)

        if response_type is True:
            # Confirmed - proceed to datetime collection
            self.confirmation_attempts['appointment_type'] = 0
            self.state = AppointmentState.COLLECT_DATETIME
            return "Got it! When would you like to schedule this appointment? You can say something like 'next Tuesday at 2pm' or 'tomorrow morning'."

        elif response_type is False:
            # Rejected - go back to appointment type selection
            self.data.appointment_type = None
            self.confirmation_attempts['appointment_type'] = 0
            self.state = AppointmentState.COLLECT_APPOINTMENT_TYPE

            types = APPOINTMENT_TYPES[self.data.department]
            types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(types)])
            return f"No worries! What type of appointment do you need?\n\n{types_list}\n\nFeel free to say the number or the appointment type."

        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['appointment_type'] = self.confirmation_attempts.get('appointment_type', 0) + 1

            if self.confirmation_attempts['appointment_type'] >= self.config.max_confirmation_attempts:
                # Default to yes after max empty attempts
                if self.config.log_state_transitions:
                    logger.info(f"Appointment type confirmation: defaulting to 'yes' after {self.confirmation_attempts['appointment_type']} empty responses")
                self.confirmation_attempts['appointment_type'] = 0
                self.state = AppointmentState.COLLECT_DATETIME
                return "Sounds good! When would you like to schedule this appointment? You can say something like 'next Tuesday at 2pm' or 'tomorrow morning'."
            else:
                # Repeat confirmation
                return f"So you need {self.data.appointment_type}, am I right? Please say 'yes' or 'no'."

    def _handle_datetime_collection(self, user_input: str) -> str:
        """Collect preferred date and time"""
        # Parse datetime from user input
        parsed_datetime = parse_datetime(user_input)

        if parsed_datetime:
            # Validate datetime
            is_valid, error_msg = validate_datetime(parsed_datetime)

            if not is_valid:
                # Validation failed - provide friendly hint
                self.retry_counts['datetime'] = self.retry_counts.get('datetime', 0) + 1

                if self.retry_counts['datetime'] < self.config.max_retries:
                    return f"{error_msg} Could you suggest another date and time?"
                else:
                    # After max retries, suggest a valid default
                    from datetime import timedelta
                    next_week = datetime.now() + timedelta(days=7)
                    default_datetime = next_week.strftime("%A, %B %d, %Y at 10:00 AM")
                    self.data.preferred_datetime = default_datetime
                    self.retry_counts['datetime'] = 0
                    self.state = AppointmentState.CONFIRM_DATETIME
                    return f"Let's try {default_datetime} instead - we can adjust later if needed. So the appointment is for {self.data.preferred_datetime}, am I right?"

            # Success path - datetime is valid
            self.data.preferred_datetime = parsed_datetime
            self.retry_counts['datetime'] = 0
            self.state = AppointmentState.CONFIRM_DATETIME

            # Transition to datetime confirmation
            return f"Perfect! So the appointment is for {parsed_datetime}, am I right?"
        else:
            # Failure path - could not parse
            self.retry_counts['datetime'] = self.retry_counts.get('datetime', 0) + 1

            if self.retry_counts['datetime'] < self.config.max_retries:
                return "I didn't quite understand that date/time. Could you try again? For example, 'next Tuesday at 2pm' or 'tomorrow morning'. Take your time!"
            else:
                # Suggest default
                from datetime import timedelta
                next_week = datetime.now() + timedelta(days=7)
                default_datetime = next_week.strftime("%A, %B %d, %Y at 10:00 AM")
                self.data.preferred_datetime = default_datetime
                self.retry_counts['datetime'] = 0
                self.state = AppointmentState.CONFIRM_DATETIME
                return f"How about {default_datetime}? We can work out the exact time later. So the appointment is for {self.data.preferred_datetime}, am I right?"

    def _handle_datetime_confirmation(self, user_input: str) -> str:
        """Handle confirmation of selected datetime"""
        response_type = parse_yes_no(user_input)

        if response_type is True:
            # Confirmed - proceed to purpose collection
            self.confirmation_attempts['datetime'] = 0
            self.state = AppointmentState.COLLECT_PURPOSE

            response = "Perfect!"
            # Add department-specific advance notice hint if available
            response += " Last question: what's the main reason for this appointment? Just a brief description is fine—no need to write an essay!"
            return response

        elif response_type is False:
            # Rejected - go back to datetime collection
            self.data.preferred_datetime = None
            self.confirmation_attempts['datetime'] = 0
            self.state = AppointmentState.COLLECT_DATETIME
            return "No problem! When would you like to schedule this appointment? You can say something like 'next Tuesday at 2pm' or 'tomorrow morning'."

        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['datetime'] = self.confirmation_attempts.get('datetime', 0) + 1

            if self.confirmation_attempts['datetime'] >= self.config.max_confirmation_attempts:
                # Default to yes after max empty attempts
                if self.config.log_state_transitions:
                    logger.info(f"Datetime confirmation: defaulting to 'yes' after {self.confirmation_attempts['datetime']} empty responses")
                self.confirmation_attempts['datetime'] = 0
                self.state = AppointmentState.COLLECT_PURPOSE
                response = "Perfect!"
                response += " Last question: what's the main reason for this appointment? Just a brief description is fine!"
                return response
            else:
                # Repeat confirmation
                return f"So the appointment is for {self.data.preferred_datetime}, am I right? Please say 'yes' or 'no'."

    def _handle_purpose_collection(self, user_input: str) -> str:
        """Collect appointment purpose/reason"""
        # Clean input
        cleaned = user_input.strip()

        # Remove common phrases
        prefixes = ["i need", "i want to", "the purpose is", "it's for", "because"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        # Validate purpose
        if len(cleaned) < 2:
            self.retry_counts['purpose'] = self.retry_counts.get('purpose', 0) + 1
            if self.retry_counts['purpose'] < self.config.max_retries:
                return "Could you tell me a bit more about why you need this appointment? For example, 'I need help choosing courses for next semester'. No worries if you need to think for a moment!"
            else:
                cleaned = "General inquiry"
        elif cleaned.lower() in ["test", "none", "n/a", ".", "help"]:
            self.retry_counts['purpose'] = self.retry_counts.get('purpose', 0) + 1
            if self.retry_counts['purpose'] < self.config.max_retries:
                return "Could you be a bit more specific about what you need help with? Feel free to give me any details that come to mind!"
            else:
                cleaned = "General inquiry"
        elif len(cleaned) > 500:
            cleaned = cleaned[:500]

        # Success path
        self.data.purpose = cleaned
        self.retry_counts['purpose'] = 0
        self.state = AppointmentState.CONFIRM_PURPOSE

        # Transition to purpose confirmation
        purpose_preview = cleaned[:50] + "..." if len(cleaned) > 50 else cleaned
        return f"Got it! So the reason for your appointment is '{purpose_preview}', am I right?"

    def _handle_purpose_confirmation(self, user_input: str) -> str:
        """Handle confirmation of appointment purpose"""
        response_type = parse_yes_no(user_input)

        if response_type is True:
            # Confirmed - proceed to final confirmation
            self.confirmation_attempts['purpose'] = 0
            self.state = AppointmentState.CONFIRM

            # Generate final confirmation summary
            summary = self._generate_confirmation_summary()
            return f"Great! Let me confirm all the details:\n\n{summary}\n\nDoes everything look correct? Say 'yes' to confirm or 'no' to make changes."

        elif response_type is False:
            # Rejected - go back to purpose collection
            self.data.purpose = None
            self.confirmation_attempts['purpose'] = 0
            self.state = AppointmentState.COLLECT_PURPOSE
            return "No worries! What's the main reason for this appointment? Feel free to give me any details!"

        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['purpose'] = self.confirmation_attempts.get('purpose', 0) + 1

            if self.confirmation_attempts['purpose'] >= self.config.max_confirmation_attempts:
                # Default to yes after max empty attempts
                if self.config.log_state_transitions:
                    logger.info(f"Purpose confirmation: defaulting to 'yes' after {self.confirmation_attempts['purpose']} empty responses")
                self.confirmation_attempts['purpose'] = 0
                self.state = AppointmentState.CONFIRM
                summary = self._generate_confirmation_summary()
                return f"Great! Let me confirm all the details:\n\n{summary}\n\nDoes everything look correct? Say 'yes' to confirm or 'no' to make changes."
            else:
                # Repeat confirmation
                purpose_preview = self.data.purpose[:50] + "..." if len(self.data.purpose) > 50 else self.data.purpose
                return f"So the reason for your appointment is '{purpose_preview}', am I right? Please say 'yes' or 'no'."

    def _handle_confirmation(self, user_input: str) -> str:
        """Handle confirmation of collected information"""
        # Clean input
        cleaned = user_input.strip().lower()

        # Handle empty responses - default to yes after max attempts
        if not cleaned:
            self.confirmation_attempts['final'] = self.confirmation_attempts.get('final', 0) + 1

            if self.confirmation_attempts['final'] >= self.config.max_confirmation_attempts:
                # Default to yes after max empty attempts
                if self.config.log_state_transitions:
                    logger.info(f"Final confirmation: defaulting to 'yes' after {self.confirmation_attempts['final']} empty responses")
                self.confirmation_attempts['final'] = 0

                # Proceed with booking completion
                self.data.booking_timestamp = datetime.now().isoformat()
                self.state = AppointmentState.COMPLETE

                response = (
                    f"Awesome! Your appointment is all set. Here's what happens next:\n\n"
                    f"1. You'll receive a confirmation email at {self.data.email} within 24 hours\n"
                    f"2. The email will include a calendar invite and any preparation instructions\n"
                    f"3. If you need to cancel or reschedule, use the link in the confirmation email\n\n"
                    f"Remember to bring your student ID and any relevant documents to the appointment.\n\n"
                    f"Is there anything else I can help you with?"
                )

                return response
            else:
                # Ask again
                return "I didn't catch that. Is all the information correct? Please say 'yes' to confirm or 'no' to make changes."

        # Reset confirmation attempts on valid input
        self.confirmation_attempts['final'] = 0

        # Detect confirmation
        if any(word in cleaned for word in ["yes", "yeah", "yep", "correct", "right", "looks good", "confirm", "ok", "okay"]):
            # Confirmation path
            self.data.booking_timestamp = datetime.now().isoformat()
            self.state = AppointmentState.COMPLETE

            response = (
                f"Awesome! Your appointment is all set. Here's what happens next:\n\n"
                f"1. You'll receive a confirmation email at {self.data.email} within 24 hours\n"
                f"2. The email will include a calendar invite and any preparation instructions\n"
                f"3. If you need to cancel or reschedule, use the link in the confirmation email\n\n"
                f"Remember to bring your student ID and any relevant documents to the appointment.\n\n"
                f"Is there anything else I can help you with?"
            )

            return response

        # Detect rejection
        elif any(word in cleaned for word in ["no", "nope", "wrong", "incorrect", "change"]):
            # Rejection path - ask which field to change
            return "No problem! Which information would you like to change? You can say 'name', 'email', 'phone', 'department', 'appointment type', 'date', or 'purpose'."

        # Detect specific field changes
        elif "name" in cleaned:
            self.data.name = None
            self.state = AppointmentState.COLLECT_NAME
            self.retry_counts['name'] = 0
            return "Okay, let's update your name. What's your full name?"
        elif "email" in cleaned:
            self.data.email = None
            self.state = AppointmentState.COLLECT_EMAIL
            self.retry_counts['email'] = 0
            return "Okay, let's update your email. What's your email address?"
        elif "phone" in cleaned:
            self.data.phone = None
            self.state = AppointmentState.COLLECT_PHONE
            self.retry_counts['phone'] = 0
            return "Okay, let's update your phone number. What's your phone number?"
        elif "department" in cleaned:
            self.data.department = None
            self.state = AppointmentState.COLLECT_DEPARTMENT
            self.retry_counts['department'] = 0
            dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
            return f"Okay, let's change the department. Which department would you like?\n\n{dept_list}"
        elif "type" in cleaned or "appointment" in cleaned:
            if self.data.department is None:
                # Department not set, collect it first
                self.data.department = None
                self.state = AppointmentState.COLLECT_DEPARTMENT
                self.retry_counts['department'] = 0
                dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
                return f"First, let's choose the department:\n\n{dept_list}\n\nYou can say the number or the department name."
            else:
                # Department is set, proceed to change appointment type
                self.data.appointment_type = None
                self.state = AppointmentState.COLLECT_APPOINTMENT_TYPE
                self.retry_counts['appointment_type'] = 0
                types = APPOINTMENT_TYPES[self.data.department]
                types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(types)])
                return f"Okay, let's change the appointment type. Which type do you need?\n\n{types_list}"
        elif "date" in cleaned or "time" in cleaned or "when" in cleaned:
            self.data.preferred_datetime = None
            self.state = AppointmentState.COLLECT_DATETIME
            self.retry_counts['datetime'] = 0
            return "Okay, let's update the date and time. When would you like to schedule this appointment?"
        elif "purpose" in cleaned or "reason" in cleaned:
            self.data.purpose = None
            self.state = AppointmentState.COLLECT_PURPOSE
            self.retry_counts['purpose'] = 0
            return "Okay, let's update the purpose. What's the main reason for this appointment?"
        else:
            # Ambiguous input
            return "I didn't catch that. Is the information correct? Please say 'yes' to confirm or 'no' to make changes."

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _generate_confirmation_summary(self) -> str:
        """Generate human-readable confirmation summary"""
        summary = (
            f"Name: {self.data.name}\n"
            f"Email: {self.data.email}\n"
            f"Phone: {self.data.phone}\n"
            f"Department: {DEPARTMENTS.get(self.data.department, self.data.department)}\n"
            f"Appointment Type: {self.data.appointment_type}\n"
            f"Preferred Date/Time: {self.data.preferred_datetime}\n"
            f"Purpose: {self.data.purpose}"
        )

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Serialize FSM state to dictionary for Redis storage"""
        return {
            'state': self.state.value,
            'data': {
                'name': self.data.name,
                'email': self.data.email,
                'phone': self.data.phone,
                'department': self.data.department,
                'appointment_type': self.data.appointment_type,
                'preferred_datetime': self.data.preferred_datetime,
                'purpose': self.data.purpose,
                'student_id': self.data.student_id,
                'preferred_language': self.data.preferred_language,
                'booking_timestamp': self.data.booking_timestamp
            },
            'retry_counts': self.retry_counts,
            'confirmation_attempts': self.confirmation_attempts,
            'last_error': self.last_error,
            'conversation_history': self.conversation_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: "AppointmentConfig") -> "AppointmentFSMManager":
        """Deserialize FSM state from dictionary"""
        # Create new instance
        instance = cls(config)

        # Restore state
        instance.state = AppointmentState(data['state'])

        # Restore data
        data_dict = data['data']
        instance.data = AppointmentData(
            name=data_dict.get('name'),
            email=data_dict.get('email'),
            phone=data_dict.get('phone'),
            department=data_dict.get('department'),
            appointment_type=data_dict.get('appointment_type'),
            preferred_datetime=data_dict.get('preferred_datetime'),
            purpose=data_dict.get('purpose'),
            student_id=data_dict.get('student_id'),
            preferred_language=data_dict.get('preferred_language', 'English'),
            booking_timestamp=data_dict.get('booking_timestamp')
        )

        # Restore tracking
        instance.retry_counts = data.get('retry_counts', {})
        instance.confirmation_attempts = data.get('confirmation_attempts', {})
        instance.last_error = data.get('last_error')
        instance.conversation_history = data.get('conversation_history', [])

        return instance

    def reset(self):
        """Reset FSM to initial state"""
        self.state = AppointmentState.INIT
        self.data = AppointmentData()
        self.retry_counts = {}
        self.confirmation_attempts = {}
        self.conversation_history = []
        self.last_error = None
        if self.config.log_state_transitions:
            logger.info(" AppointmentFSMManager reset")

    def get_current_prompt(self) -> str:
        """
        Get the current prompt to speak based on FSM state.
        
        Useful for resuming after a RAG detour or other interruptions.
        
        Returns:
            The prompt that should be spoken for the current state.
        """
        if self.state == AppointmentState.INIT:
            return "Let's continue with your appointment booking. What's your full name?"
        
        elif self.state == AppointmentState.COLLECT_NAME:
            return "What's your full name?"
        
        elif self.state == AppointmentState.CONFIRM_NAME:
            spelled_name = spell_out_name(self.data.name) if self.data.name else "your name"
            return f"So your name is {spelled_name}, am I right?"
        
        elif self.state == AppointmentState.COLLECT_EMAIL:
            return "What's your email address?"
        
        elif self.state == AppointmentState.CONFIRM_EMAIL:
            return f"So your email is {self.data.email}, am I right?"
        
        elif self.state == AppointmentState.COLLECT_PHONE:
            return "What's your phone number? Include the country code if calling from outside Germany."
        
        elif self.state == AppointmentState.CONFIRM_PHONE:
            formatted_phone = format_phone_for_readback(self.data.phone) if self.data.phone else "your phone number"
            return f"So your phone number is {formatted_phone}, am I right?"
        
        elif self.state == AppointmentState.COLLECT_DEPARTMENT:
            dept_list = ", ".join(DEPARTMENTS.values())
            return f"Which department would you like to meet with? Options include: {dept_list}"
        
        elif self.state == AppointmentState.CONFIRM_DEPARTMENT:
            dept_name = DEPARTMENTS.get(self.data.department, self.data.department)
            return f"So you want to meet with {dept_name}, am I right?"
        
        elif self.state == AppointmentState.COLLECT_APPOINTMENT_TYPE:
            if self.data.department and self.data.department in APPOINTMENT_TYPES:
                types = APPOINTMENT_TYPES[self.data.department]
                types_list = ", ".join(types[:3])  # Show first 3
                return f"What type of appointment do you need? Options include: {types_list}"
            return "What type of appointment do you need?"
        
        elif self.state == AppointmentState.CONFIRM_APPOINTMENT_TYPE:
            return f"So you need {self.data.appointment_type}, am I right?"
        
        elif self.state == AppointmentState.COLLECT_DATETIME:
            return "When would you like to schedule this appointment? You can say something like 'next Tuesday at 2pm'."
        
        elif self.state == AppointmentState.CONFIRM_DATETIME:
            return f"So the appointment is for {self.data.preferred_datetime}, am I right?"
        
        elif self.state == AppointmentState.COLLECT_PURPOSE:
            return "What's the main reason for this appointment? Just a brief description is fine."
        
        elif self.state == AppointmentState.CONFIRM_PURPOSE:
            purpose_preview = self.data.purpose[:50] + "..." if self.data.purpose and len(self.data.purpose) > 50 else self.data.purpose
            return f"So the reason is '{purpose_preview}', am I right?"
        
        elif self.state == AppointmentState.CONFIRM:
            return "Does everything look correct? Say 'yes' to confirm or 'no' to make changes."
        
        elif self.state == AppointmentState.COMPLETE:
            return "Your appointment is already booked!"
        
        elif self.state == AppointmentState.CANCELLED:
            return "The booking was cancelled. Would you like to start over?"
        
        else:
            return "Let's continue with your appointment booking."