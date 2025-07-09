from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.files.base import ContentFile
from django.db import IntegrityError
from .models import Student, Attendance, EmotionLog
import base64
import io
from PIL import Image
from datetime import date, datetime

# Create your views here.

def home(request):
    # Calculate real statistics from database
    total_students = Student.objects.count()
    today = date.today()
    
    # Get today's attendance records
    today_attendance = Attendance.objects.filter(date=today)
    present_today = today_attendance.filter(status='present').count()
    late_today = today_attendance.filter(status='late').count()
    absent_today = total_students - present_today - late_today
    
    context = {
        'total_students': total_students,
        'present_today': present_today,
        'late_today': late_today,
        'absent_today': absent_today,
    }
    return render(request, "attendance/home.html", context)

def register(request):
    if request.method == 'POST':
        try:
            # Extract form data
            name = request.POST.get('name')
            roll_number = request.POST.get('roll_number')
            email = request.POST.get('email')
            phone = request.POST.get('phone', '')
            department = request.POST.get('department')
            
            # Handle image - either uploaded file or webcam capture
            image_file = request.FILES.get('image')
            webcam_image = request.POST.get('webcam_image')
            
            # Create student instance
            student = Student(
                name=name,
                roll_number=roll_number,
                email=email,
                phone=phone,
                department=department
            )
            
            # Process image
            if image_file:
                student.image = image_file
            elif webcam_image:
                # Process base64 webcam image
                format, imgstr = webcam_image.split(';base64,')
                ext = format.split('/')[-1]
                image_data = ContentFile(base64.b64decode(imgstr), name=f'{roll_number}_webcam.{ext}')
                student.image = image_data
            else:
                messages.error(request, 'Please provide an image either by upload or webcam capture.')
                return render(request, "attendance/register.html")
            
            # Save student
            student.save()
            messages.success(request, f'Student {name} registered successfully!')
            return redirect('register')
            
        except IntegrityError as e:
            if 'roll_number' in str(e):
                messages.error(request, 'A student with this roll number already exists.')
            elif 'email' in str(e):
                messages.error(request, 'A student with this email already exists.')
            else:
                messages.error(request, 'An error occurred during registration.')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
    
    return render(request, "attendance/register.html")

def scan(request):
    return render(request, "attendance/scan.html")

def students(request):
    # Get all students with search functionality
    search_query = request.GET.get('search', '')
    department_filter = request.GET.get('department', '')
    
    students = Student.objects.all()
    
    if search_query:
        students = students.filter(
            name__icontains=search_query
        ) | students.filter(
            roll_number__icontains=search_query
        )
    
    if department_filter:
        students = students.filter(department=department_filter)
    
    # Get unique departments for filter dropdown
    departments = Student.objects.values_list('department', flat=True).distinct()
    
    context = {
        'students': students,
        'departments': departments,
        'search_query': search_query,
        'department_filter': department_filter,
    }
    return render(request, "attendance/students.html", context)

def attendance(request):
    # Get attendance records with date filtering
    selected_date = request.GET.get('date', date.today().strftime('%Y-%m-%d'))
    
    try:
        filter_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
    except ValueError:
        filter_date = date.today()
    
    attendance_records = Attendance.objects.filter(date=filter_date).select_related('student')
    
    # Calculate statistics for the selected date
    total_students = Student.objects.count()
    present_count = attendance_records.filter(status='present').count()
    late_count = attendance_records.filter(status='late').count()
    absent_count = total_students - present_count - late_count
    
    context = {
        'attendance_records': attendance_records,
        'selected_date': selected_date,
        'present_count': present_count,
        'late_count': late_count,
        'absent_count': absent_count,
        'total_students': total_students,
    }
    return render(request, "attendance/attendance.html", context)

# API Views
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
import json

def student_detail_api(request, student_id):
    """API endpoint to get student details"""
    student = get_object_or_404(Student, id=student_id)
    
    data = {
        'id': student.id,
        'name': student.name,
        'roll_number': student.roll_number,
        'email': student.email,
        'phone': student.phone,
        'department': student.department,
        'image': student.image.url if student.image else None,
        'created_at': student.created_at.isoformat(),
    }
    
    return JsonResponse(data)

@csrf_exempt
def student_delete_api(request, student_id):
    """API endpoint to delete a student"""
    if request.method == 'POST':
        try:
            student = get_object_or_404(Student, id=student_id)
            student_name = student.name
            student.delete()
            
            return JsonResponse({
                'success': True,
                'message': f'Student {student_name} deleted successfully'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Error deleting student: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})

def attendance_emotions_api(request, record_id):
    """API endpoint to get emotion records for an attendance record"""
    attendance_record = get_object_or_404(Attendance, id=record_id)
    emotions = EmotionLog.objects.filter(attendance=attendance_record).order_by('-timestamp')
    
    emotions_data = []
    for emotion in emotions:
        emotions_data.append({
            'emotion': emotion.emotion,
            'confidence': emotion.confidence,
            'timestamp': emotion.timestamp.isoformat(),
        })
    
    data = {
        'student_name': attendance_record.student.name,
        'date': attendance_record.date.strftime('%Y-%m-%d'),
        'emotions': emotions_data
    }
    
    return JsonResponse(data)
