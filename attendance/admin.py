from django.contrib import admin
from .models import Student, Attendance, EmotionLog

# Register your models here.

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['name', 'roll_number', 'email', 'department', 'created_at']
    list_filter = ['department', 'created_at']
    search_fields = ['name', 'roll_number', 'email']
    ordering = ['name']

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'date', 'check_in_time', 'check_out_time', 'status', 'confidence']
    list_filter = ['status', 'date', 'student__department']
    search_fields = ['student__name', 'student__roll_number']
    date_hierarchy = 'date'
    ordering = ['-date', 'student__name']

@admin.register(EmotionLog)
class EmotionLogAdmin(admin.ModelAdmin):
    list_display = ['student', 'emotion', 'confidence', 'timestamp']
    list_filter = ['emotion', 'timestamp', 'student__department']
    search_fields = ['student__name', 'student__roll_number']
    date_hierarchy = 'timestamp'
    ordering = ['-timestamp']
