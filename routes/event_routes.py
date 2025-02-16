from flask import Blueprint, request, jsonify
from services.gemini_service import GeminiService
from services.calendar_service import CalendarService
from config import Config

events = Blueprint('events', __name__)

gemini_service = GeminiService(Config.GOOGLE_CLOUD_PROJECT_ID)
calendar_service = CalendarService(Config.GOOGLE_APPLICATION_CREDENTIALS)


@events.route('/api/events', methods=['POST'])
def create_event():
    try:
        user_input = request.json.get('user_input')
        if not user_input:
            return jsonify({'error': 'No user input provided'}), 400

        # Parse event details using Gemini
        event = gemini_service.parse_event_details(user_input)

        # Create calendar event
        calendar_event = calendar_service.create_event(event)

        return jsonify({
            'status': 'success',
            'event_link': calendar_event.get('htmlLink')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500