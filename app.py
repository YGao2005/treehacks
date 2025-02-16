from flask import Flask, request, jsonify
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

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
                {"role": "system", "content": "You are a JSON generator that returns only valid, complete JSON arrays with no additional text."},
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)