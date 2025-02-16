from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from models.event import Event


class CalendarService:
    def __init__(self, credentials_path):
        self.creds = Credentials.from_authorized_user_file(
            credentials_path,
            ['https://www.googleapis.com/auth/calendar']
        )
        self.service = build('calendar', 'v3', credentials=self.creds)

    def create_event(self, event: Event):
        return self.service.events().insert(
            calendarId='primary',
            body=event.to_calendar_event()
        ).execute()