from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from datetime import time, timedelta
import logging

logger = logging.getLogger(__name__)

# Create your models here.

class Department(models.Model):
    """Department model for organizing students and courses"""
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=10, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.code})"
    
    class Meta:
        ordering = ['name']


class Student(models.Model):
    """Student model with authentication integration"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True, related_name='student_profile')
    name = models.CharField(max_length=100)
    roll_number = models.CharField(max_length=20, unique=True, null=True, blank=True)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, blank=True)
    department = models.CharField(max_length=100)  # Legacy field, kept for backward compatibility
    department_ref = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True, related_name='students')
    image = models.ImageField(upload_to='student_images/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Additional fields
    enrollment_year = models.IntegerField(null=True, blank=True)
    graduation_year = models.IntegerField(null=True, blank=True)
    address = models.TextField(blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    is_verified = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.name} ({self.roll_number})"

    class Meta:
        ordering = ['name']


class Course(models.Model):
    """Course model for organizing classes"""
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=20, unique=True)
    description = models.TextField(blank=True)
    department = models.ForeignKey(Department, on_delete=models.CASCADE, related_name='courses')
    instructor = models.CharField(max_length=100)
    semester = models.IntegerField(default=1)
    credits = models.IntegerField(default=3)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.code}: {self.name}"
    
    class Meta:
        ordering = ['department', 'code']


class ClassSchedule(models.Model):
    """Schedule for course classes"""
    DAY_CHOICES = [
        (0, 'Monday'),
        (1, 'Tuesday'),
        (2, 'Wednesday'),
        (3, 'Thursday'),
        (4, 'Friday'),
        (5, 'Saturday'),
        (6, 'Sunday'),
    ]
    
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='schedules')
    day_of_week = models.IntegerField(choices=DAY_CHOICES)
    start_time = models.TimeField()
    end_time = models.TimeField()
    room = models.CharField(max_length=50)
    
    def __str__(self):
        return f"{self.course.code} - {self.get_day_of_week_display()} {self.start_time.strftime('%H:%M')}"
    
    class Meta:
        ordering = ['day_of_week', 'start_time']
        unique_together = ['course', 'day_of_week', 'start_time']


class CourseEnrollment(models.Model):
    """Students enrolled in courses"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='enrollments')
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='enrollments')
    enrollment_date = models.DateField(default=timezone.now)
    active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.student.roll_number} - {self.course.code}"
    
    class Meta:
        unique_together = ['student', 'course']


class Attendance(models.Model):
    """Attendance records for students"""
    ATTENDANCE_STATUS_CHOICES = [
        ('present', 'Present'),
        ('late', 'Late'),
        ('absent', 'Absent'),
    ]
    
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='attendance_records')
    course = models.ForeignKey(Course, on_delete=models.SET_NULL, null=True, blank=True, related_name='attendance_records')
    date = models.DateField(default=timezone.now)
    check_in_time = models.TimeField(null=True, blank=True)
    check_out_time = models.TimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=ATTENDANCE_STATUS_CHOICES, default='absent')
    confidence = models.FloatField(default=0.0)  # Face recognition confidence
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
  
        if self.check_in_time and not kwargs.get('force_status', False):
            # Get attendance settings
            settings = AttendanceSettings.get_settings()
            
            if self.check_in_time <= settings.on_time_threshold:
                self.status = 'present'
            elif self.check_in_time <= settings.late_threshold:
                self.status = 'late'
            else:
                self.status = 'absent'
                
        if kwargs.get('force_status', False):
            kwargs.pop('force_status')
            
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.student.name} - {self.date} - {self.status}"

    class Meta:
        unique_together = ['student', 'date', 'course']
        ordering = ['-date', 'student__name']




class AttendanceSettings(models.Model):
    """System-wide attendance settings"""
    on_time_threshold = models.TimeField(default=time(9, 10)) 
    late_threshold = models.TimeField(default=time(10, 0))    
    absent_threshold = models.TimeField(default=time(10, 30))  
    min_attendance_percentage = models.FloatField(default=75.0)
    face_recognition_threshold = models.FloatField(default=0.9) 
    allow_late_marking = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    @classmethod
    def get_settings(cls):
        """Get the system settings, creating default if none exist"""
        settings, created = cls.objects.get_or_create(pk=1)
        return settings
    
    def __str__(self):
        return f"Attendance Settings (Updated: {self.updated_at.strftime('%Y-%m-%d')})"
    
    class Meta:
        verbose_name = "Attendance Settings"
        verbose_name_plural = "Attendance Settings"


class Notification(models.Model):
    """Notifications for students and admins"""
    NOTIFICATION_TYPES = [
        ('attendance', 'Attendance Update'),
        ('warning', 'Attendance Warning'),
        ('system', 'System Notification'),
        ('course', 'Course Update'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    title = models.CharField(max_length=100)
    message = models.TextField()
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"
    
    class Meta:
        ordering = ['-created_at']


class AttendanceReport(models.Model):
    """Generated attendance reports"""
    REPORT_TYPES = [
        ('daily', 'Daily Report'),
        ('weekly', 'Weekly Report'),
        ('monthly', 'Monthly Report'),
        ('custom', 'Custom Report'),
    ]
    
    title = models.CharField(max_length=100)
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES)
    start_date = models.DateField()
    end_date = models.DateField()
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)
    course = models.ForeignKey(Course, on_delete=models.SET_NULL, null=True, blank=True)
    generated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='generated_reports')
    report_file = models.FileField(upload_to='reports/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.title} - {self.report_type} ({self.start_date} to {self.end_date})"
    
    class Meta:
        ordering = ['-created_at']


class StudentPerformance(models.Model):
    """Student performance prediction data"""
    PERFORMANCE_CATEGORIES = [
        ('excellent', 'Excellent (90-100)'),
        ('very_good', 'Very Good (80-89)'),
        ('good', 'Good (70-79)'),
        ('satisfactory', 'Satisfactory (60-69)'),
        ('needs_improvement', 'Needs Improvement (50-59)'),
        ('poor', 'Poor (Below 50)'),
    ]
    
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='performance_predictions')
    
    # Input parameters (0-100 scale)
    attendance_score = models.FloatField(
        help_text="Attendance percentage (0-100)",
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    previous_grades = models.FloatField(
        help_text="Previous grades average (0-100)",
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    assignment_completion = models.FloatField(
        help_text="Assignment completion rate (0-100)",
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    class_activeness = models.FloatField(
        help_text="Class participation and activeness (0-100)",
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    
    # Weights for weighted average (should sum to 100)
    attendance_weight = models.FloatField(default=30.0, help_text="Weight for attendance (default 30%)")
    grades_weight = models.FloatField(default=40.0, help_text="Weight for previous grades (default 40%)")
    assignment_weight = models.FloatField(default=20.0, help_text="Weight for assignments (default 20%)")
    activeness_weight = models.FloatField(default=10.0, help_text="Weight for class activeness (default 10%)")
    
    # Calculated results
    overall_score = models.FloatField(editable=False)
    performance_category = models.CharField(max_length=20, choices=PERFORMANCE_CATEGORIES, editable=False)
    
    # Metadata
    predicted_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='performance_predictions')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        """Calculate overall score and performance category before saving"""
        # Calculate weighted average
        total_weight = self.attendance_weight + self.grades_weight + self.assignment_weight + self.activeness_weight
        
        if total_weight > 0:
            self.overall_score = (
                (self.attendance_score * self.attendance_weight) +
                (self.previous_grades * self.grades_weight) +
                (self.assignment_completion * self.assignment_weight) +
                (self.class_activeness * self.activeness_weight)
            ) / total_weight
        else:
            self.overall_score = 0
        
        # Determine performance category
        if self.overall_score >= 90:
            self.performance_category = 'excellent'
        elif self.overall_score >= 80:
            self.performance_category = 'very_good'
        elif self.overall_score >= 70:
            self.performance_category = 'good'
        elif self.overall_score >= 60:
            self.performance_category = 'satisfactory'
        elif self.overall_score >= 50:
            self.performance_category = 'needs_improvement'
        else:
            self.performance_category = 'poor'
        
        super().save(*args, **kwargs)
    
    def get_category_display_with_color(self):
        """Get performance category with appropriate CSS class"""
        category_colors = {
            'excellent': 'success',
            'very_good': 'info',
            'good': 'primary',
            'satisfactory': 'warning',
            'needs_improvement': 'warning',
            'poor': 'danger',
        }
        return {
            'category': self.get_performance_category_display(),
            'color': category_colors.get(self.performance_category, 'secondary')
        }
    
    def __str__(self):
        return f"{self.student.name} - {self.overall_score:.1f}% ({self.get_performance_category_display()})"
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['student'] 


