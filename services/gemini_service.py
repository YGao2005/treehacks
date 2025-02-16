from google.cloud import aiplatform
from models.event import Event


class GeminiService:
    def __init__(self, project_id, location="us-central1"):
        aiplatform.init(project=project_id, location=location)
        self.model = aiplatform.GenerativeModel('gemini-pro')

    def parse_event_details(self, user_input: str) -> Event:
        prompt = """
        Extract event details from the following text. Return a JSON with:
        - title: event title
        - start_time: start datetime (ISO format)
        - end_time: end datetime (ISO format)
        - description: event description

        Text: {user_input}
        """

        response = self.model.generate_content(prompt.format(user_input=user_input))
        # Parse response and create Event object
        # Add error handling as needed
        return Event(**response.text)