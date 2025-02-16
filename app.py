from flask import Flask, request, jsonify
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from google.cloud import aiplatform
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google import generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
from datetime import datetime, timedelta
from google.oauth2 import service_account

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

aiplatform.init(project="your-project-id", location="us-central1")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

# Initialize Calendar service
def get_calendar_service():
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    creds = None

    if os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        except (EOFError, pickle.UnpicklingError):
            os.remove('token.pickle')
            print("Removed corrupted token file")

    if not creds or not creds.valid:
        try:
            creds = Credentials(
                token="ya29.a0AXeO80QRkuLEL6yzGflFy3-JwuQt6RAHoi_fWnNjYqCAuV1_LWqeM7iJZGh_hbwm5VU0LiBMNojpR2GIevW83xghfOdkMBgPSG8Hvguae9Lp9210sVv3Uzjr1Vp89qYNQP14HudJ5reR2CGu102kTLWytwNr6Ya9deKp6rYJaCgYKAfkSARESFQHGX2MivusCrF_dNJK1MCEvjpA-eQ0175",
                refresh_token="1//04luznsSLvZ16CgYIARAAGAQSNwF-L9IrL-m8CTXmx-2C6_1HiE39IWwaGfvOzA0MXqC-HysSsWWI9INhzfd4b_XZSkIGUZ28YmU",
                token_uri="https://oauth2.googleapis.com/token",
                client_id="43249297252-10cvponppasks9p2cunjrttdvq92qbj0.apps.googleusercontent.com",
                client_secret="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",
                scopes=SCOPES
            )

            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
            print("Created and saved new token file")

        except Exception as e:
            print(f"Error creating credentials: {str(e)}")
            raise

    return build('calendar', 'v3', credentials=creds)

calendar_service = get_calendar_service()

activity_df = pd.read_csv("activity_type.csv")
destress_activities = set(activity_df[activity_df['Classification'] == 'De-stressor']['Activity'].str.lower())

app = Flask(__name__)

@app.route('/add_destresser_to_calendar', methods=['POST'])
def add_destresser_to_calendar():
    try:
        # Get the destresser recommendations and date/time input
        destresser_data = request.json.get('destresser_data')
        date_time_input = request.json.get('date_time')

        if not destresser_data or not date_time_input:
            return jsonify({'error': 'Missing destresser_data or date_time input'}), 400

        # Parse the date and time
        try:
            event_datetime = datetime.strptime(date_time_input, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            return jsonify({'error': 'Invalid date_time format. Use ISO format: YYYY-MM-DDTHH:MM:SS'}), 400

        # Add each destresser place as an event
        for destresser in destresser_data:
            place = destresser['place']
            activities = ", ".join(destresser['activities'])

            event = {
                'summary': f'Destresser: {place}',
                'description': f"Activities: {activities}",
                'start': {
                    'dateTime': event_datetime.isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
                'end': {
                    'dateTime': (event_datetime + timedelta(hours=1)).isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
            }

            calendar_service.events().insert(calendarId='primary', body=event).execute()

        return jsonify({'status': 'success', 'message': 'Destresser activities added to calendar'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_workout_to_calendar', methods=['POST'])
def add_workout_to_calendar():
    try:
        workout_plan = request.json
        if not workout_plan:
            return jsonify({'error': 'No workout plan provided'}), 400

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        for workout in workout_plan:
            day_of_week = workout['day_of_week']
            time_of_day = workout['time_of_day']
            workout_regimen = workout['workout_regimen']

            # Convert day_of_week and time_of_day to a datetime object
            today = datetime.today()
            days_ahead = (days.index(day_of_week) - today.weekday() + 7) % 7
            workout_date = today + timedelta(days=days_ahead)
            workout_datetime = datetime.strptime(f"{workout_date.strftime('%Y-%m-%d')} {time_of_day}", '%Y-%m-%d %I:%M %p')

            event = {
                'summary': 'Workout',
                'description': workout_regimen,
                'start': {
                    'dateTime': workout_datetime.isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
                'end': {
                    'dateTime': (workout_datetime + timedelta(hours=1)).isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
            }

            calendar_service.events().insert(calendarId='primary', body=event).execute()

        return jsonify({'status': 'success', 'message': 'Workout plan added to calendar'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create-event', methods=['POST'])
def create_event():
    try:
        # Validate input
        user_input = request.json.get('user_input')
        if not user_input:
            return jsonify({'error': 'No user input provided'}), 400

        # Use Gemini to parse event details
        prompt = f'''
        Generate a precise JSON event object for: {user_input}

        Return a JSON object in this exact format:
        {{
        "title": "Event Title",
        "start_time": "2024-02-16T14:00:00-08:00",
        "end_time": "2024-02-16T15:00:00-08:00",
        "description": "Event description"
        }}

        STRICT REQUIREMENTS:
        - Times must be explicitly in Pacific Standard Time (PST, UTC-08:00) with ISO 8601 format and timezone offset (-08:00).
        - Include date, month, and year accurately in PST.
        - Derive the end_time by correctly adding the duration to the start_time.
        - Times must include seconds (:00) for both start_time and end_time.
        - Automatically calculate tomorrow’s date from the current PST date if specified.
        - Return only the JSON object without any extra text, headers, or explanations.
        '''

        response = gemini_model.generate_content(prompt)

        # Clean the response
        response_text = response.text.strip()
        response_text = response_text.replace('json', '').replace('\n', '').strip()
        response_text = ' '.join(response_text.split())  # Remove newlines

        # Extract JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            return jsonify({
                'error': 'Could not find valid JSON in response',
                'raw_response': response_text
            }), 500

        json_str = response_text[json_start:json_end]

        # Parse JSON
        event_details = json.loads(json_str)

        # Validate fields
        required_fields = ['title', 'start_time', 'end_time', 'description']
        missing_fields = [field for field in required_fields if field not in event_details]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'received_fields': list(event_details.keys())
            }), 500

        # Create Calendar event
        event = {
            'summary': event_details['title'],
            'description': event_details['description'],
            'start': {
                'dateTime': event_details['start_time'],
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': event_details['end_time'],
                'timeZone': 'UTC',
            },
        }

        # Create the calendar event
        calendar_event = calendar_service.events().insert(
            calendarId='primary',
            body=event
        ).execute()

        return jsonify({
            'status': 'success',
            'event_details': event_details,
            'event_link': calendar_event.get('htmlLink')
        })

    except json.JSONDecodeError as e:
        return jsonify({
            'error': 'Failed to parse JSON response',
            'details': str(e),
            'raw_response': response_text
        }), 500
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@app.route('/get_destresser_recommendations', methods=['POST'])
def get_destresser_recommendations():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Expecting JSON payload."}), 400

        request_data = request.get_json()

        filtered_prompt = (
                "Return a list of 8 suitable places near Stanford University as a JSON array. "
                "Each element should have a 'place' and 'activities' properties. "
                "The activities should be chosen from this list: " +
                ", ".join(list(destress_activities)[:40]) +
                ". Return ONLY the JSON array with no additional text or explanation."
        )

        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system",
                 "content": "You are a JSON generator that returns only valid, complete JSON arrays with no additional text."},
                {"role": "user", "content": filtered_prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        content = response.choices[0].message.content.strip()

        if '[' in content:
            content = content[content.index('['):]
            if not content.endswith(']'):
                content = content.rstrip(',') + '\n]'

        try:
            recommendations = json.loads(content)

            with open("stanford_destress_recommendations.json", "w") as f:
                json.dump(recommendations, f, indent=2)

            return jsonify(recommendations)
        except json.JSONDecodeError as e:
            print("JSON parsing error:", str(e))
            print("Cleaned content:", content)

            try:
                import re
                objects = re.findall(r'\{[^{}]*\}(?=\s*,|\s*\])', content)
                if objects:
                    valid_json = '[' + ','.join(objects) + ']'
                    recommendations = json.loads(valid_json)
                    return jsonify(recommendations)
            except:
                pass

            return jsonify({
                "error": "Could not parse response as JSON",
                "raw_response": content
            }), 500

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/get_workout_plan', methods=['POST'])
def get_workout_plan():
    try:
        file_path = 'payload.txt'
        if not os.path.exists(file_path):
            return jsonify({"error": "File 'payload.txt' not found. Please ensure the file is uploaded."}), 400

        with open(file_path, 'r') as file:
            biometric_data = file.read()

        prompt = (
            "You are a JSON generator that outputs only valid JSON arrays without any additional explanation.\n"
            "DO NOT include any text, summaries, or introductions outside of the JSON array.\n"
            "DO NOT explain your reasoning.\n"
            "DO NOT add any extra details.\n\n"
            "Analyze the following biometric data to create a 4-day outdoor workout plan:\n"
            f"{biometric_data}\n\n"
            "Requirements:\n"
            "- Search the web for the best outdoor workout times (excluding 9 AM to 5 PM).\n"
            "- Format the output as a JSON array with exactly 4 objects.\n"
            "- Each object must contain only these fields: 'day_of_week', 'time_of_day', and 'workout_regimen'.\n"
            "- Use common time formats (e.g., '6:00 AM').\n"
            "- Use simple day names ('Monday', 'Tuesday', etc.).\n\n"
            "Return ONLY a valid JSON array with no additional text or explanations."
        )

        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "Return only valid JSON arrays with no extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        response_content = response.choices[0].message.content.strip()
        print(f"Raw API Response: {response_content}")

        workout_plan = json.loads(response.choices[0].message.content.strip())
        return jsonify(workout_plan)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)