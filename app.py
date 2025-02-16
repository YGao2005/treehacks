from flask import Flask, request, jsonify
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from google import genai
from google.cloud import aiplatform
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google import generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os

from datetime import datetime
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

    # Try to load existing credentials
    if os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        except (EOFError, pickle.UnpicklingError):
            # If pickle file is corrupted or empty, remove it
            os.remove('token.pickle')
            print("Removed corrupted token file")

    # If credentials don't exist or are invalid
    if not creds or not creds.valid:
        # Create new credentials
        try:
            creds = Credentials(
                token="ya29.a0AXeO80QRkuLEL6yzGflFy3-JwuQt6RAHoi_fWnNjYqCAuV1_LWqeM7iJZGh_hbwm5VU0LiBMNojpR2GIevW83xghfOdkMBgPSG8Hvguae9Lp9210sVv3Uzjr1Vp89qYNQP14HudJ5reR2CGu102kTLWytwNr6Ya9deKp6rYJaCgYKAfkSARESFQHGX2MivusCrF_dNJK1MCEvjpA-eQ0175",
                refresh_token="1//04luznsSLvZ16CgYIARAAGAQSNwF-L9IrL-m8CTXmx-2C6_1HiE39IWwaGfvOzA0MXqC-HysSsWWI9INhzfd4b_XZSkIGUZ28YmU",
                token_uri="https://oauth2.googleapis.com/token",
                client_id="43249297252-10cvponppasks9p2cunjrttdvq92qbj0.apps.googleusercontent.com",
                client_secret="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",
                scopes=SCOPES
            )

            # Save new credentials
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
            print("Created and saved new token file")

        except Exception as e:
            print(f"Error creating credentials: {str(e)}")
            raise

    return build('calendar', 'v3', credentials=creds)

# Replace the original calendar_service initialization with this
calendar_service = get_calendar_service()

activity_df = pd.read_csv("activity_type.csv")
destress_activities = set(activity_df[activity_df['Type'] == 'destresser']['Activity'].str.lower())

app = Flask(__name__)


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
        file_path = 'feb1.txt'
        if not os.path.exists(file_path):
            return jsonify({"error": "File 'feb1.txt' not found. Please ensure the file is uploaded."}), 400

        with open(file_path, 'r') as file:
            biometric_data = file.read()

        prompt = (
            f"Analyze this biometric data: \n{biometric_data}\n"
            "Search the web for the best times to work out outdoors (excluding 9 AM to 5 PM). "
            "Generate a 4-day workout plan in JSON with only 'day_of_week', 'time_of_day', and 'workout_regimen'."
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

        workout_plan = json.loads(response.choices[0].message.content.strip())
        return jsonify(workout_plan)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/create-event', methods=['POST'])
def create_event():
    try:
        # Validate input
        user_input = request.json.get('user_input')
        if not user_input:
            return jsonify({'error': 'No user input provided'}), 400

        # Use Gemini to parse event details
        prompt = f'''Generate a JSON event object for: {user_input}

Return a single JSON object with these fields using this format:
{{"title": "Event Title", "start_time": "2024-02-16T14:00:00+00:00", "end_time": "2024-02-16T15:00:00+00:00", "description": "Event description"}}

Remember:
- Use tomorrow's date where needed
- Add duration to start_time to get end_time
- Use PST timezone (UTC-07:00)
- Include seconds as :00'''

        # Get response from Gemini
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        print(f"Raw Gemini response: {response_text}")  # Debug log

        # Clean the response
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        response_text = ' '.join(response_text.split())  # Remove newlines
        print(f"Cleaned response: {response_text}")  # Debug log

        # Extract JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            return jsonify({
                'error': 'Could not find valid JSON in response',
                'raw_response': response_text
            }), 500

        json_str = response_text[json_start:json_end]
        print(f"Extracted JSON string: {json_str}")  # Debug log

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

        # First, try to get the calendar ACL to check if your email is already added
        your_email = "psdyangg@gmail.com"  # Replace with your actual email

        try:
            # Create the calendar event
            calendar_event = calendar_service.events().insert(
                calendarId='primary',
                body=event
            ).execute()

            # Share the calendar with your personal account if not already shared
            rule = {
                'role': 'writer',
                'scope': {
                    'type': 'user',
                    'value': your_email
                }
            }

            calendar_service.acl().insert(
                calendarId='primary',
                body=rule
            ).execute()

        except Exception as calendar_error:
            print(f"Calendar sharing error: {str(calendar_error)}")
            # Continue with the response even if sharing fails
            pass

        return jsonify({
            'status': 'success',
            'event_details': event_details,
            'event_link': calendar_event.get('htmlLink')
        })

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")  # Debug log
        print(f"Attempted to parse: {json_str}")  # Debug log
        return jsonify({
            'error': 'Failed to parse JSON response',
            'details': str(e),
            'raw_response': response_text
        }), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Debug log
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
