from django.utils import timezone
from django.db.models import Count, Q
from datetime import datetime, timedelta, date
import calendar
from .models import Student, Attendance, Course, Department, AttendanceSettings

def get_attendance_stats(student=None, course=None, start_date=None, end_date=None):
    """
    Get attendance statistics for a student, course, or overall.
    
    Args:
        student: Optional Student object to filter by
        course: Optional Course object to filter by
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        Dictionary with attendance statistics
    """
    # Start with all attendance records
    queryset = Attendance.objects.all()
    
    # Apply filters
    if student:
        queryset = queryset.filter(student=student)
    
    if course:
        queryset = queryset.filter(course=course)
    
    if start_date:
        queryset = queryset.filter(date__gte=start_date)
    
    if end_date:
        queryset = queryset.filter(date__lte=end_date)
    
    # Count by status
    present_count = queryset.filter(status='present').count()
    late_count = queryset.filter(status='late').count()
    absent_count = queryset.filter(status='absent').count()
    
    # Calculate total and rate
    total_count = present_count + late_count + absent_count
    
    if total_count > 0:
        # Count late as half present for rate calculation
        attendance_rate = round(((present_count + (late_count * 0.5)) / total_count) * 100)
    else:
        attendance_rate = 0
    
    return {
        'present': present_count,
        'late': late_count,
        'absent': absent_count,
        'total': total_count,
        'rate': attendance_rate
    }

def get_monthly_attendance_data(year=None, month=None, department=None):
    """
    Get attendance data for a specific month, grouped by day.
    
    Args:
        year: Year (defaults to current year)
        month: Month number (defaults to current month)
        department: Optional Department object to filter by
        
    Returns:
        Dictionary with daily attendance counts
    """
    if not year:
        year = timezone.now().year
    
    if not month:
        month = timezone.now().month
    
    # Get the number of days in the month
    num_days = calendar.monthrange(year, month)[1]
    
    # Create a list of dates for the month
    dates = [date(year, month, day) for day in range(1, num_days + 1)]
    
    # Start with all attendance records for the month
    queryset = Attendance.objects.filter(date__year=year, date__month=month)
    
    # Apply department filter if provided
    if department:
        queryset = queryset.filter(student__department_ref=department)
    
    # Group by date and status
    daily_data = {}
    
    for current_date in dates:
        day_records = queryset.filter(date=current_date)
        daily_data[current_date] = {
            'present': day_records.filter(status='present').count(),
            'late': day_records.filter(status='late').count(),
            'absent': day_records.filter(status='absent').count(),
            'total': day_records.count()
        }
    
    return daily_data

def mark_attendance(student, course=None, check_in_time=None, status=None, confidence=0.0):
    """
    Mark attendance for a student.
    
    Args:
        student: Student object
        course: Optional Course object
        check_in_time: Optional check-in time (defaults to current time)
        status: Optional status override (defaults to auto-calculation based on time)
        confidence: Face recognition confidence score
        
    Returns:
        Tuple of (attendance_record, created)
    """
    current_datetime = timezone.localtime(timezone.now())
    today = current_datetime.date()
    
    if not check_in_time:
        check_in_time = current_datetime.time()
    
    # Check if attendance already exists for today
    try:
        attendance = Attendance.objects.get(student=student, date=today, course=course)
        created = False
    except Attendance.DoesNotExist:
        attendance = Attendance(student=student, date=today, course=course)
        created = True
    
    # Update check-in time if not already set
    if not attendance.check_in_time:
        attendance.check_in_time = check_in_time
    
    # Set confidence score
    attendance.confidence = max(attendance.confidence, confidence)
    
    # Override status if provided
    if status:
        attendance.save(force_status=True)
        attendance.status = status
    
    attendance.save()
    
    return attendance, created

def get_student_attendance_summary(student, start_date=None, end_date=None):
    """
    Get a summary of attendance for a student.
    
    Args:
        student: Student object
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        Dictionary with attendance summary
    """
    # Default to the current academic year
    current_datetime = timezone.localtime(timezone.now())
    if not start_date:
        today = current_datetime.date()
        # Academic year typically starts in August/September
        if today.month < 8:
            start_date = date(today.year - 1, 8, 1)
        else:
            start_date = date(today.year, 8, 1)
    
    if not end_date:
        end_date = current_datetime.date()
    
    # Get all attendance records for the student in the date range
    attendance_records = Attendance.objects.filter(
        student=student,
        date__gte=start_date,
        date__lte=end_date
    )
    
    # Get stats by course
    courses = Course.objects.filter(attendance_records__student=student).distinct()
    course_stats = {}
    
    for course in courses:
        course_records = attendance_records.filter(course=course)
        present = course_records.filter(status='present').count()
        late = course_records.filter(status='late').count()
        absent = course_records.filter(status='absent').count()
        total = present + late + absent
        
        if total > 0:
            rate = round(((present + (late * 0.5)) / total) * 100)
        else:
            rate = 0
        
        course_stats[course.id] = {
            'course': course,
            'present': present,
            'late': late,
            'absent': absent,
            'total': total,
            'rate': rate
        }
    
    # Get overall stats
    overall_stats = get_attendance_stats(student=student, start_date=start_date, end_date=end_date)
    
    # Get attendance settings for minimum required attendance
    settings = AttendanceSettings.get_settings()
    min_required = settings.min_attendance_percentage
    
    # Determine if student meets attendance requirements
    meets_requirements = overall_stats['rate'] >= min_required
    
    return {
        'student': student,
        'start_date': start_date,
        'end_date': end_date,
        'overall_stats': overall_stats,
        'course_stats': course_stats,
        'min_required': min_required,
        'meets_requirements': meets_requirements
    }

def generate_attendance_report(start_date, end_date, department=None, course=None, report_type='custom'):
    """
    Generate attendance report data.
    
    Args:
        start_date: Start date for the report
        end_date: End date for the report
        department: Optional Department object to filter by
        course: Optional Course object to filter by
        report_type: Report type (daily, weekly, monthly, custom)
        
    Returns:
        Dictionary with report data
    """
    # Start with all attendance records in the date range
    queryset = Attendance.objects.filter(date__gte=start_date, date__lte=end_date)
    
    # Apply filters
    if department:
        queryset = queryset.filter(student__department_ref=department)
    
    if course:
        queryset = queryset.filter(course=course)
    
    # Get overall stats
    present_count = queryset.filter(status='present').count()
    late_count = queryset.filter(status='late').count()
    absent_count = queryset.filter(status='absent').count()
    total_count = present_count + late_count + absent_count
    
    if total_count > 0:
        attendance_rate = round(((present_count + (late_count * 0.5)) / total_count) * 100)
    else:
        attendance_rate = 0
    
    # Get stats by date
    dates = queryset.values('date').distinct().order_by('date')
    daily_stats = {}
    
    for date_dict in dates:
        current_date = date_dict['date']
        day_records = queryset.filter(date=current_date)
        
        daily_stats[current_date] = {
            'present': day_records.filter(status='present').count(),
            'late': day_records.filter(status='late').count(),
            'absent': day_records.filter(status='absent').count(),
            'total': day_records.count()
        }
    
    # Get stats by student
    students = Student.objects.filter(attendance_records__in=queryset).distinct()
    student_stats = {}
    
    for student in students:
        student_records = queryset.filter(student=student)
        present = student_records.filter(status='present').count()
        late = student_records.filter(status='late').count()
        absent = student_records.filter(status='absent').count()
        total = present + late + absent
        
        if total > 0:
            rate = round(((present + (late * 0.5)) / total) * 100)
        else:
            rate = 0
        
        student_stats[student.id] = {
            'student': student,
            'present': present,
            'late': late,
            'absent': absent,
            'total': total,
            'rate': rate
        }
    
    # Get stats by course (if applicable)
    course_stats = {}
    
    if not course:  # Only if not already filtered by a specific course
        courses = Course.objects.filter(attendance_records__in=queryset).distinct()
        
        for course_obj in courses:
            course_records = queryset.filter(course=course_obj)
            present = course_records.filter(status='present').count()
            late = course_records.filter(status='late').count()
            absent = course_records.filter(status='absent').count()
            total = present + late + absent
            
            if total > 0:
                rate = round(((present + (late * 0.5)) / total) * 100)
            else:
                rate = 0
            
            course_stats[course_obj.id] = {
                'course': course_obj,
                'present': present,
                'late': late,
                'absent': absent,
                'total': total,
                'rate': rate
            }
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'department': department,
        'course': course,
        'report_type': report_type,
        'overall_stats': {
            'present': present_count,
            'late': late_count,
            'absent': absent_count,
            'total': total_count,
            'rate': attendance_rate
        },
        'daily_stats': daily_stats,
        'student_stats': student_stats,
        'course_stats': course_stats
    } 