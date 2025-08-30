from django.core.management.base import BaseCommand
from attendance.utils.training_analytics import record_training_event


class Command(BaseCommand):
    help = "Stub: train face detector (e.g., MTCNN)."

    def handle(self, *args, **options):
        start = record_training_event('train_mtcnn_started')
        self.stdout.write(self.style.SUCCESS(f"Detector training started: {start}"))
        done = record_training_event('train_mtcnn_completed', {"status": "ok"})
        self.stdout.write(self.style.SUCCESS(f"Detector training completed: {done}"))


