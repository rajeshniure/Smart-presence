from django.core.management.base import BaseCommand
from attendance.utils.training_analytics import record_training_event


class Command(BaseCommand):
    help = "Stub: train a simple face recognition classifier."

    def handle(self, *args, **options):
        start = record_training_event('train_simple_recognition_started')
        self.stdout.write(self.style.SUCCESS(f"Recognition training started: {start}"))
        done = record_training_event('train_simple_recognition_completed', {"status": "ok"})
        self.stdout.write(self.style.SUCCESS(f"Recognition training completed: {done}"))


