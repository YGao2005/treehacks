from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    title: str
    start_time: datetime
    end_time: datetime
    description: str = ""

    def to_calendar_event(self):
        return {
            'summary': self.title,
            'description': self.description,
            'start': {
                'dateTime': self.start_time.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': self.end_time.isoformat(),
                'timeZone': 'UTC',
            },
        }