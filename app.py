from flask import Flask, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")

client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

app = Flask(__name__)

@app.route('/get_physical_recommendations', methods=['POST'])
def get_physical_recommendations():
    data = request.json
    disease = data.get('disease')
    health_characteristics = data.get('health_characteristics')

    if not disease or not health_characteristics:
        return jsonify({"error": "Missing 'disease' or 'health_characteristics'"}), 400

    prompt = (
        "You are a health expert. Provide exactly 5 concise physical health recommendations in pure JSON format.\n"
        "Each recommendation should be under 20 words.\n"
        "Output only the JSON object with the following structure:\n"
        "{\"recommendations\":[{\"id\":1,\"advice\":\"...\"}, ...]}\n"
        "Ensure that the output contains ONLY valid JSON with no markdown formatting, code fences, or additional text.\n"
        f"Context:\n- Disease: {disease}\n- Characteristics: {health_characteristics}\n"
    )

    try:
        response = client.chat.completions.create(
            model="sonar-reasoning-pro",
            messages=[
                {"role": "system", "content": "Return only valid JSON with no markdown or extra formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        print("API Response:", repr(content))

        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start == -1 or json_end == -1:
            return jsonify({"error": "Could not extract JSON from API response"}), 500

        json_content = content[json_start:json_end+1]

        try:
            recommendations_json = json.loads(json_content)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Failed to parse JSON: {str(e)}", "raw_content": json_content}), 500

        if "recommendations" not in recommendations_json:
            return jsonify({"error": "Missing 'recommendations' field in response"}), 500

        # Optionally, save the recommendations to a file
        with open("physical_recommendations.json", "w") as json_file:
            json.dump(recommendations_json, json_file)

        return jsonify(recommendations_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)