from django.contrib import admin
from .models import (
    Student, Attendance, Department, Course, 
    ClassSchedule, CourseEnrollment, AttendanceSettings,
    Notification, AttendanceReport, StudentPerformance
)

# Register your models here.

@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'created_at']
    search_fields = ['name', 'code']
    ordering = ['name']

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ['code', 'name', 'department', 'instructor', 'semester', 'credits']
    list_filter = ['department', 'semester', 'credits']
    search_fields = ['name', 'code', 'instructor']
    ordering = ['department', 'code']

@admin.register(ClassSchedule)
class ClassScheduleAdmin(admin.ModelAdmin):
    list_display = ['course', 'day_of_week', 'start_time', 'end_time', 'room']
    list_filter = ['day_of_week', 'course__department']
    search_fields = ['course__name', 'course__code', 'room']
    ordering = ['day_of_week', 'start_time']

@admin.register(CourseEnrollment)
class CourseEnrollmentAdmin(admin.ModelAdmin):
    list_display = ['student', 'course', 'enrollment_date', 'active']
    list_filter = ['active', 'enrollment_date', 'course__department']
    search_fields = ['student__name', 'student__roll_number', 'course__name', 'course__code']
    ordering = ['-enrollment_date']

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['name', 'roll_number', 'email', 'department', 'created_at']
    list_filter = ['department', 'created_at']
    search_fields = ['name', 'roll_number', 'email']
    ordering = ['name']
    fieldsets = (
        (None, {
            'fields': ('user', 'name', 'roll_number', 'email')
        }),
        ('Department Information', {
            'fields': ('department', 'department_ref')
        }),
        ('Personal Information', {
            'fields': ('phone', 'address', 'date_of_birth', 'image')
        }),
        ('Academic Information', {
            'fields': ('enrollment_year', 'graduation_year')
        }),
    )
    
    def delete_model(self, request, obj):
        """Override delete_model to also delete the associated user account"""
        if obj.user:
            user_username = obj.user.username
            obj.user.delete()
            self.message_user(request, f"User account '{user_username}' has also been deleted.")
        obj.delete()
    
    def delete_queryset(self, request, queryset):
        """Override delete_queryset to also delete associated user accounts when bulk deleting"""
        deleted_users = []
        for obj in queryset:
            if obj.user:
                deleted_users.append(obj.user.username)
                obj.user.delete()
        
        if deleted_users:
            self.message_user(request, f"User accounts for {', '.join(deleted_users)} have also been deleted.")
        
        queryset.delete()
    
    def retrain_model(self, request, queryset):
        """Custom action to retrain the FaceNet model"""
        from django.core.management import call_command
        import threading
        
        def retrain_async():
            try:
                call_command('retrain_facenet', '--force', verbosity=1)
                from attendance.utils.face_pipeline import reload_models
                reload_models()
            except Exception as e:
                print(f"Error in retraining: {str(e)}")
        
        # Start retraining in background thread
        thread = threading.Thread(target=retrain_async)
        thread.daemon = True
        thread.start()
        
        self.message_user(request, "FaceNet model retraining started in background. This may take a few minutes.")
    
    retrain_model.short_description = "Retrain FaceNet Model"
    
    actions = ['retrain_model']

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'course', 'date', 'check_in_time', 'check_out_time', 'status', 'confidence']
    list_filter = ['status', 'date', 'student__department', 'course']
    search_fields = ['student__name', 'student__roll_number', 'course__name', 'course__code']
    date_hierarchy = 'date'
    ordering = ['-date', 'student__name']


@admin.register(AttendanceSettings)
class AttendanceSettingsAdmin(admin.ModelAdmin):
    list_display = ['on_time_threshold', 'late_threshold', 'absent_threshold', 'min_attendance_percentage', 'face_recognition_threshold', 'updated_at']
    fieldsets = (
        ('Attendance Time Settings', {
            'fields': ('on_time_threshold', 'late_threshold', 'absent_threshold')
        }),
        ('System Settings', {
            'fields': ('min_attendance_percentage', 'face_recognition_threshold', 'allow_late_marking'),
            'description': 'Face recognition threshold: Distance threshold (0.5-1.2). Lower = stricter matching. Recommended: 0.8-1.0'
        }),
    )
    
    def has_add_permission(self, request):
        # Only allow one settings object
        return AttendanceSettings.objects.count() == 0

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'notification_type', 'is_read', 'created_at']
    list_filter = ['notification_type', 'is_read', 'created_at']
    search_fields = ['title', 'message', 'user__username']
    ordering = ['-created_at']

@admin.register(AttendanceReport)
class AttendanceReportAdmin(admin.ModelAdmin):
    list_display = ['title', 'report_type', 'start_date', 'end_date', 'department', 'course', 'created_at']
    list_filter = ['report_type', 'department', 'course', 'created_at']
    search_fields = ['title', 'department__name', 'course__name']
    date_hierarchy = 'created_at'
    ordering = ['-created_at']


@admin.register(StudentPerformance)
class StudentPerformanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'overall_score', 'performance_category', 'predicted_by', 'created_at']
    list_filter = ['performance_category', 'created_at', 'student__department']
    search_fields = ['student__name', 'student__roll_number', 'predicted_by__username']
    readonly_fields = ['overall_score', 'performance_category']
    date_hierarchy = 'created_at'
    ordering = ['-created_at']
    
    fieldsets = (
        ('Student Information', {
            'fields': ('student', 'predicted_by')
        }),
        ('Performance Metrics (0-100)', {
            'fields': ('attendance_score', 'previous_grades', 'assignment_completion', 'class_activeness'),
            'description': 'Enter values between 0 and 100 for each performance metric.'
        }),
        ('Weights Configuration (%)', {
            'fields': ('attendance_weight', 'grades_weight', 'assignment_weight', 'activeness_weight'),
            'description': 'Weights for the weighted average calculation. Should sum to 100%.'
        }),
        ('Calculated Results', {
            'fields': ('overall_score', 'performance_category'),
            'description': 'These fields are automatically calculated based on the metrics and weights above.'
        }),
    )
    
    def save_model(self, request, obj, form, change):
        """Set the predicted_by field to the current user if not set"""
        if not obj.predicted_by:
            obj.predicted_by = request.user
        super().save_model(request, obj, form, change)
