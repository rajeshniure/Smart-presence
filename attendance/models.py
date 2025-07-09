from django.db import models
from django.utils import timezone
from datetime import time

# Create your models here.

class Student(models.Model):
    name = models.CharField(max_length=100)
    roll_number = models.CharField(max_length=20, unique=True)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, blank=True)
    department = models.CharField(max_length=100)
    image = models.ImageField(upload_to='student_images/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.roll_number})"

    class Meta:
        ordering = ['name']


class Attendance(models.Model):
    ATTENDANCE_STATUS_CHOICES = [
        ('present', 'Present'),
        ('late', 'Late'),
        ('absent', 'Absent'),
    ]
    
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    check_in_time = models.TimeField(null=True, blank=True)
    check_out_time = models.TimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=ATTENDANCE_STATUS_CHOICES, default='absent')
    confidence = models.FloatField(default=0.0)  # Face recognition confidence
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        # Auto-determine status based on check-in time
        if self.check_in_time:
            if self.check_in_time <= time(9, 10):  # 9:10 AM
                self.status = 'present'
            elif self.check_in_time <= time(10, 0):  # 10:00 AM
                self.status = 'late'
            else:
                self.status = 'absent'
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.student.name} - {self.date} - {self.status}"

    class Meta:
        unique_together = ['student', 'date']
        ordering = ['-date', 'student__name']


class EmotionLog(models.Model):
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('angry', 'Angry'),
        ('neutral', 'Neutral'),
        ('disgusted', 'Disgusted'),
        ('fearful', 'Fearful'),
        ('surprised', 'Surprised'),
    ]

    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    attendance = models.ForeignKey(Attendance, on_delete=models.CASCADE, related_name='emotions')
    emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    confidence = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.name} - {self.emotion} ({self.confidence:.2f})"

    class Meta:
        ordering = ['-timestamp']
