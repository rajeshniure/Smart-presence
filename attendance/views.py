from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.paginator import Paginator
from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.db.models import Count
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student, Attendance, Course, StudentPerformance
import base64
import json
from datetime import date, datetime
import calendar

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status as http_status
from django.utils import timezone
from .utils.face_pipeline import detect_and_recognize
import os
from django.core.files.storage import default_storage
import logging


# Set up logging
logger = logging.getLogger(__name__)

# Helper functions
def is_admin(user):
    return user.is_staff or user.is_superuser

def is_student(user):
    return hasattr(user, 'student_profile') and user.student_profile is not None

def get_or_create_student_image(image_file, webcam_image, filename_prefix):
    if image_file:
        return image_file
    if webcam_image:
        format, imgstr = webcam_image.split(';base64,')
        ext = format.split('/')[-1]
        return ContentFile(base64.b64decode(imgstr), name=f'{filename_prefix}_webcam.{ext}')
    return None

def get_attendance_stats(qs):
    total = qs.count()
    present = qs.filter(status='present').count()
    late = qs.filter(status='late').count()
    absent = qs.filter(status='absent').count()
    rate = round(((present + (late * 0.5)) / total) * 100) if total > 0 else 0
    return {'present': present, 'late': late, 'absent': absent, 'rate': rate}

def get_unverified_count(request):
    if request.user.is_authenticated and is_admin(request.user):
        return Student.objects.filter(is_verified=False).count()
    return 0

# Landing page view
def landing(request):
    return render(request, "landing_page.html")

# Authentication views
def admin_login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None and (user.is_staff or user.is_superuser):
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('home')
        else:
            messages.error(request, "Invalid credentials or insufficient permissions.")
            return redirect('landing')
    
    return redirect('landing')

def student_login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        try:
            student = Student.objects.get(email=email)
            if not student.user:
                messages.error(request, "This student is not registered for login. Please contact your teacher.")
                return redirect('landing')
            if not student.is_verified:
                messages.error(request, "Your registration is pending verification by a teacher/admin. Please wait for approval.")
                return redirect('landing')
            user = authenticate(request, username=student.user.username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {student.name}!")
                return redirect('student_home')
            messages.error(request, "Invalid credentials.")
        except Student.DoesNotExist:
            messages.error(request, "Student with this email does not exist.")
        
        return redirect('landing')
    
    return redirect('landing')

def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('landing')

# Admin views
@login_required
@user_passes_test(is_admin)
def home(request):
    # Calculate real statistics from database
    from .models import AttendanceSettings
    # Handle AttendanceSettings updates from home page
    if request.method == 'POST' and request.POST.get('form_type') == 'thresholds':
        try:
            from django.utils.dateparse import parse_time
            on_time_threshold = parse_time(request.POST.get('on_time_threshold') or '')
            late_threshold = parse_time(request.POST.get('late_threshold') or '')
            absent_threshold = parse_time(request.POST.get('absent_threshold') or '')
            if not (on_time_threshold and late_threshold and absent_threshold):
                messages.error(request, 'All three thresholds are required.')
                return redirect('home')
            if not (on_time_threshold <= late_threshold <= absent_threshold):
                messages.error(request, 'Invalid order: On-time must be ≤ Late ≤ Absent.')
                return redirect('home')
            settings_obj = AttendanceSettings.get_settings()
            settings_obj.on_time_threshold = on_time_threshold
            settings_obj.late_threshold = late_threshold
            settings_obj.absent_threshold = absent_threshold
            settings_obj.save()
            messages.success(request, 'Attendance thresholds updated successfully.')
            return redirect('home')
        except Exception as e:
            messages.error(request, f'Failed to update thresholds: {str(e)}')
            return redirect('home')
    total_students = Student.objects.count()
    today = date.today()
    
    # Get today's attendance records
    today_attendance = Attendance.objects.filter(date=today)
    present_today = today_attendance.filter(status='present').count()
    late_today = today_attendance.filter(status='late').count()
    absent_today = total_students - present_today - late_today
    
    settings_obj = AttendanceSettings.get_settings()
    context = {
        'total_students': total_students,
        'present_today': present_today,
        'late_today': late_today,
        'absent_today': absent_today,
        'unverified_count': get_unverified_count(request),
        'settings_on_time_threshold': settings_obj.on_time_threshold.strftime('%H:%M') if settings_obj.on_time_threshold else '',
        'settings_late_threshold': settings_obj.late_threshold.strftime('%H:%M') if settings_obj.late_threshold else '',
        'settings_absent_threshold': settings_obj.absent_threshold.strftime('%H:%M') if settings_obj.absent_threshold else '',
    }
    return render(request, "teacherDashboard/home.html", context)


def register(request):
    if request.method == 'POST':
        try:
            # Extract form data
            name = request.POST.get('name')
            email = request.POST.get('email')
            phone = request.POST.get('phone', '')
            department = request.POST.get('department')
            password = request.POST.get('password', '')
            confirm_password = request.POST.get('confirm_password', '')
            
            # Enforce password requirement
            if not password:
                messages.error(request, 'Password is required for student login access.')
                return render(request, "register.html")
            # Confirm password match
            if password != confirm_password:
                messages.error(request, 'Passwords do not match.')
                return render(request, "register.html")
            
            # Handle image - either uploaded file or webcam capture
            image_file = request.FILES.get('image')
            webcam_image = request.POST.get('webcam_image')
            
            # Process image
            filename_prefix = (email.split('@')[0] if email and '@' in email else 'student')
            student_image = get_or_create_student_image(image_file, webcam_image, filename_prefix)
            if not student_image:
                messages.error(request, 'Please provide an image either by upload or webcam capture.')
                return render(request, "register.html")
            
            # Save image temporarily for face validation
            import tempfile
            import os
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
                    for chunk in student_image.chunks() if hasattr(student_image, 'chunks') else [student_image.read()]:
                        temp_img.write(chunk)
                    temp_path = temp_img.name
                # Validate face
                faces = detect_and_recognize(temp_path)
                if not faces or len(faces) == 0:
                    messages.error(request, 'The uploaded image must contain a clear human face. Please try again.')
                    os.remove(temp_path)
                    return render(request, "register.html")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Create student instance
            student = Student(
                name=name,
                email=email,
                phone=phone,
                department=department,
                is_verified=False
            )
            student.image = student_image
            
            # Create user account
            username = email  # Use email as username; roll number will be set on verification
            user = User.objects.create_user(username=username, email=email, password=password)
            user.first_name = name.split()[0] if ' ' in name else name
            user.last_name = name.split(' ', 1)[1] if ' ' in name else ''
            user.save()
            student.user = user
            
            # Save student
            student.save()
            messages.success(request, f'Registration successful! Please wait for verification by a teacher/admin before logging in.')
            return redirect('register')
            
        except IntegrityError as e:
            if 'email' in str(e):
                messages.error(request, 'A student with this email already exists.')
            else:
                messages.error(request, f'An error occurred during registration: {str(e)}')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
    
    return render(request, "register.html")

def scan(request):
    if request.method == 'POST' and request.FILES.get('image'):
        abs_path = handle_uploaded_image(request, 'temp_scan.jpg')
        results = recognize_and_cleanup(abs_path)
        # Ensure 'Unknown' faces are not matched to students
        for face in results:
            name = face.get('name', '')
            if name.lower() == 'unknown':
                face['roll_number'] = None
                face['department'] = None
                face['photo_url'] = None
        return render(request, 'scan.html', {'results': results})
    return render(request, 'scan.html')

@login_required
@user_passes_test(is_admin)
def students(request):
    # Get all students with search functionality
    search_query = request.GET.get('search', '')
    department_filter = request.GET.get('department', '')
    
    students = Student.objects.filter(is_verified=True)
    
    if search_query:
        students = students.filter(
            name__icontains=search_query
        ) | students.filter(
            roll_number__icontains=search_query
        )
    
    if department_filter:
        students = students.filter(department=department_filter)
    
    # Get unique departments for filter dropdown
    departments = (
        Student.objects
        .exclude(department__isnull=True)
        .exclude(department__exact="")
        .values_list('department', flat=True)
        .distinct()
        .order_by('department')
    )
    
    context = {
        'students': students,
        'departments': departments,
        'search_query': search_query,
        'department_filter': department_filter,
        'unverified_count': get_unverified_count(request),
    }
    return render(request, "teacherDashboard/students.html", context)

@login_required
@user_passes_test(is_admin)
def attendance(request):
    # Handle AttendanceSettings updates
    from .models import AttendanceSettings
    if request.method == 'POST' and request.POST.get('form_type') == 'thresholds':
        try:
            from django.utils.dateparse import parse_time
            on_time_threshold = parse_time(request.POST.get('on_time_threshold') or '')
            late_threshold = parse_time(request.POST.get('late_threshold') or '')
            absent_threshold = parse_time(request.POST.get('absent_threshold') or '')
            if not (on_time_threshold and late_threshold and absent_threshold):
                messages.error(request, 'All three thresholds are required.')
                return redirect('attendance')
            # Validate ordering: on_time <= late <= absent
            if not (on_time_threshold <= late_threshold <= absent_threshold):
                messages.error(request, 'Invalid order: On-time must be ≤ Late ≤ Absent.')
                return redirect('attendance')
            settings_obj = AttendanceSettings.get_settings()
            settings_obj.on_time_threshold = on_time_threshold
            settings_obj.late_threshold = late_threshold
            settings_obj.absent_threshold = absent_threshold
            settings_obj.save()
            messages.success(request, 'Attendance thresholds updated successfully.')
            return redirect('attendance')
        except Exception as e:
            messages.error(request, f'Failed to update thresholds: {str(e)}')
            return redirect('attendance')

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
    
    # Load current settings to display in UI
    settings_obj = AttendanceSettings.get_settings()

    context = {
        'attendance_records': attendance_records,
        'selected_date': selected_date,
        'present_count': present_count,
        'late_count': late_count,
        'absent_count': absent_count,
        'total_students': total_students,
        'unverified_count': get_unverified_count(request),
        'settings_on_time_threshold': settings_obj.on_time_threshold.strftime('%H:%M') if settings_obj.on_time_threshold else '',
        'settings_late_threshold': settings_obj.late_threshold.strftime('%H:%M') if settings_obj.late_threshold else '',
        'settings_absent_threshold': settings_obj.absent_threshold.strftime('%H:%M') if settings_obj.absent_threshold else '',
    }
    return render(request, "teacherDashboard/attendance.html", context)

# Student Dashboard Views
@login_required
@user_passes_test(is_student)
def student_home(request):
    student = request.user.student_profile
    today = date.today()
    
    # Get today's attendance
    try:
        today_attendance = Attendance.objects.get(student=student, date=today)
    except Attendance.DoesNotExist:
        today_attendance = None
    
    # Calculate attendance statistics
    attendance_stats = get_attendance_stats(Attendance.objects.filter(student=student))
    
    context = {
        'student': student,
        'today_attendance': today_attendance,
        'attendance_stats': attendance_stats,
    }
    
    return render(request, "studentDashboard/home.html", context)

@login_required
@user_passes_test(is_student)
def student_attendance(request):
    student = request.user.student_profile
    selected_month = request.GET.get('month', '')
    
    # Get attendance records
    attendance_records = Attendance.objects.filter(student=student).order_by('-date')
    
    # Filter by month if selected
    if selected_month:
        try:
            month = int(selected_month)
            year = datetime.now().year
            attendance_records = attendance_records.filter(date__month=month, date__year=year)
        except ValueError:
            pass
    
    # Calculate attendance statistics for the filtered records
    attendance_stats = get_attendance_stats(attendance_records)
    
    # Prepare month choices for the filter
    months = [(str(i), calendar.month_name[i]) for i in range(1, 13)]
    
    # Paginate the results
    paginator = Paginator(attendance_records, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'student': student,
        'attendance_records': page_obj,
        'attendance_stats': attendance_stats,
        'months': months,
        'selected_month': selected_month,
    }
    
    return render(request, "studentDashboard/attendance.html", context)

@login_required
@user_passes_test(is_student)
def student_profile(request):
    student = request.user.student_profile
    
    context = {
        'student': student,
    }
    
    return render(request, "studentDashboard/profile.html", context)

@login_required
@user_passes_test(is_student)
def student_profile_update(request):
    if request.method == 'POST':
        student = request.user.student_profile
        user = request.user
        
        # Update email and phone
        email = request.POST.get('email')
        phone = request.POST.get('phone', '')
        
        # Check if email is already taken by another user
        if email != student.email and User.objects.filter(email=email).exclude(id=user.id).exists():
            messages.error(request, 'This email is already in use by another account.')
            return redirect('student_profile')
        
        student.email = email
        student.phone = phone
        
        # Update user email
        user.email = email
        
        # Update password if provided
        password = request.POST.get('password')
        if password:
            user.set_password(password)
            messages.info(request, 'Your password has been updated. Please login again.')
        
        user.save()
        student.save()
        
        messages.success(request, 'Your profile has been updated successfully.')
        
        # If password was changed, log the user out to force re-login
        if password:
            return redirect('logout')
        
        return redirect('student_profile')
    
    return redirect('student_profile')

@login_required
@user_passes_test(is_student)
def student_photo_update(request):
    if request.method == 'POST':
        student = request.user.student_profile
        
        # Handle image - either uploaded file or webcam capture
        image_file = request.FILES.get('image')
        webcam_image = request.POST.get('webcam_image')
        
        student.image = get_or_create_student_image(image_file, webcam_image, student.roll_number)
        
        if not student.image:
            messages.error(request, 'No image provided.')
            return redirect('student_profile')
        
        student.save()
        messages.success(request, 'Your profile photo has been updated successfully.')
        
        return redirect('student_profile')
    
    return redirect('student_profile')

# API Views
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
def student_update_api(request, student_id):
    """API endpoint to update student information"""
    if request.method == 'POST':
        try:
            student = get_object_or_404(Student, id=student_id)
            data = json.loads(request.body)
            
            # Update student fields
            student.name = data.get('name', student.name)
            student.email = data.get('email', student.email)
            student.phone = data.get('phone', student.phone)
            student.department = data.get('department', student.department)
            
            # Only update roll number if it's changed and doesn't conflict
            new_roll_number = data.get('roll_number')
            if new_roll_number and new_roll_number != student.roll_number:
                if Student.objects.filter(roll_number=new_roll_number).exclude(id=student_id).exists():
                    return JsonResponse({
                        'success': False,
                        'message': 'A student with this roll number already exists'
                    })
                student.roll_number = new_roll_number
            
            # Handle image update if provided
            webcam_image = data.get('webcam_image')
            if webcam_image and webcam_image.startswith('data:image'):
                format, imgstr = webcam_image.split(';base64,')
                ext = format.split('/')[-1]
                image_data = ContentFile(base64.b64decode(imgstr), name=f'{student.roll_number}_webcam.{ext}')
                student.image = image_data
            
            student.save()
            
            return JsonResponse({
                'success': True,
                'message': f'Student {student.name} updated successfully',
                'student': {
                    'id': student.id,
                    'name': student.name,
                    'roll_number': student.roll_number,
                    'email': student.email,
                    'phone': student.phone,
                    'department': student.department,
                    'image': student.image.url if student.image else None,
                }
            })
        except IntegrityError as e:
            if 'email' in str(e):
                return JsonResponse({
                    'success': False,
                    'message': 'A student with this email already exists'
                })
            return JsonResponse({
                'success': False,
                'message': f'Database integrity error: {str(e)}'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Error updating student: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})

@csrf_exempt
def student_delete_api(request, student_id):
    """API endpoint to delete a student and their associated user account"""
    if request.method == 'POST':
        try:
            student = get_object_or_404(Student, id=student_id)
            student_name = student.name
            
            # Delete the associated user account if it exists
            if student.user:
                user_username = student.user.username
                student.user.delete()
                logger.info(f"Deleted user account: {user_username}")
            
            # Delete the student record
            student.delete()
            
            return JsonResponse({
                'success': True,
                'message': f'Student {student_name} and their user account deleted successfully'
            })
        except Exception as e:
            logger.error(f"Error deleting student {student_id}: {str(e)}")
            return JsonResponse({
                'success': False,
                'message': f'Error deleting student: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})



@login_required
@user_passes_test(is_admin)
def attendance_export_api(request):
    """Export attendance records for a given date as CSV.
    Query params: ?date=YYYY-MM-DD (defaults to today)
    """
    try:
        export_date_str = request.GET.get('date')
        if export_date_str:
            try:
                export_date = datetime.strptime(export_date_str, '%Y-%m-%d').date()
            except ValueError:
                return JsonResponse({'success': False, 'message': 'Invalid date format. Use YYYY-MM-DD.'}, status=400)
        else:
            export_date = date.today()

        records = Attendance.objects.filter(date=export_date).select_related('student')

        import csv
        from io import StringIO
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)

        # Header
        writer.writerow([
            'Date',
            'Student Name',
            'Roll Number',
            'Department',
            'Check-in Time',
            'Check-out Time',
            'Status',
        ])

        # Rows
        for r in records:
            writer.writerow([
                export_date.strftime('%Y-%m-%d'),
                getattr(r.student, 'name', ''),
                getattr(r.student, 'roll_number', ''),
                getattr(r.student, 'department', ''),
                r.check_in_time.strftime('%H:%M') if r.check_in_time else '',
                r.check_out_time.strftime('%H:%M') if r.check_out_time else '',
                r.status,
            ])

        from django.http import HttpResponse
        response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="attendance_{export_date.strftime('%Y%m%d')}.csv"'
        return response
    except Exception as e:
        return JsonResponse({'success': False, 'message': f'Failed to export: {str(e)}'}, status=500)

@csrf_exempt
def attendance_edit_api(request, record_id):
    from django.utils.dateparse import parse_time
    att = get_object_or_404(Attendance, id=record_id)
    if request.method == 'GET':
        return JsonResponse({
            'success': True,
            'record': {
                'id': att.id,
                'status': att.status,
                'check_in_time': att.check_in_time.strftime('%H:%M') if att.check_in_time else '',
                'check_out_time': att.check_out_time.strftime('%H:%M') if att.check_out_time else '',
            }
        })
    elif request.method == 'POST':
        status = request.POST.get('status')
        check_in_time = request.POST.get('check_in_time')
        check_out_time = request.POST.get('check_out_time')
        if status in ['present','late','absent']:
            att.status = status
        att.check_in_time = parse_time(check_in_time) if check_in_time else None
        att.check_out_time = parse_time(check_out_time) if check_out_time else None
        att.save()
        return JsonResponse({'success': True, 'message': 'Attendance updated'})
    return JsonResponse({'success': False, 'message': 'Invalid request'}, status=400)


def recognize_and_cleanup(abs_path):
    """Run face recognition and clean up temp file."""
    try:
        results = detect_and_recognize(abs_path)
    finally:
        try:
            os.remove(abs_path)
        except Exception:
            pass
    return results

def handle_uploaded_image(request, temp_filename):
    """Save uploaded image to temp file and return absolute path. Raises ValueError if no image."""
    image_file = request.FILES.get('image')
    if not image_file:
        raise ValueError('No image provided.')
    temp_path = default_storage.save(temp_filename, image_file)
    abs_path = default_storage.path(temp_path)
    return abs_path

def build_student_data(face, student=None, attendance_status=None, attendance_id=None, attendance_details=None):
    data = {
        'name': face['name'],
        'box': face['box'],
        'attendance_status': attendance_status,
        'attendance_id': attendance_id,
        'attendance_details': attendance_details,
    }
    if student:
        data.update({
            'roll_number': student.roll_number,
            'department': student.department,
            'photo_url': student.image.url if student.image else None,
        })
    else:
        data.update({'roll_number': None, 'department': None, 'photo_url': None})
    return data

class ScanAPIView(APIView):
    authentication_classes = []
    permission_classes = []
    def post(self, request, *args, **kwargs):
        try:
            try:
                abs_path = handle_uploaded_image(request, 'temp_scan.jpg')
            except ValueError as e:
                return Response({'error': str(e)}, status=http_status.HTTP_400_BAD_REQUEST)
            try:
                results = recognize_and_cleanup(abs_path)
            except Exception:
                return Response({'error': 'Error in face detection/recognition.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            response_data = []
            current_datetime = timezone.localtime(timezone.now())
            today = current_datetime.date()
            for face in results:
                try:
                    name = face['name']
                    if name.lower() == 'unknown':
                        response_data.append(build_student_data(face, None, 'unknown', None, None))
                        continue
                    student = Student.objects.filter(name=name).first()
                    attendance_status = 'unknown'
                    attendance_id = None
                    attendance_details = None
                    if student:
                        current_time = current_datetime.time()
                        current_date = current_datetime.date()
                        existing_attendance = Attendance.objects.filter(
                            student=student, date=current_date, course=None
                        ).first()
                        if existing_attendance:
                            attendance_status = 'already_marked'
                            attendance_id = existing_attendance.id
                            attendance_details = {
                                'status': existing_attendance.status,
                                'check_in_time': existing_attendance.check_in_time.strftime('%H:%M') if existing_attendance.check_in_time else None,
                                'check_out_time': existing_attendance.check_out_time.strftime('%H:%M') if existing_attendance.check_out_time else None
                            }
                        else:
                            from .models import AttendanceSettings
                            settings = AttendanceSettings.get_settings()
                            if current_time <= settings.on_time_threshold:
                                attendance_status_value = 'present'
                            elif current_time <= settings.late_threshold:
                                attendance_status_value = 'late'
                            else:
                                attendance_status_value = 'absent'
                            att = Attendance.objects.create(
                                student=student,
                                date=current_date,
                                course=None,
                                check_in_time=current_time,
                                status=attendance_status_value,
                                confidence=1.0
                            )
                            attendance_status = attendance_status_value
                            attendance_id = att.id
                            attendance_details = {
                                'status': attendance_status_value,
                                'check_in_time': current_time.strftime('%H:%M'),
                                'check_out_time': None
                            }
                    response_data.append(build_student_data(face, student, attendance_status, attendance_id, attendance_details))
                except Exception:
                    continue
            return Response({'results': response_data}, status=http_status.HTTP_200_OK)
        except Exception:
            return Response({'error': 'Internal server error.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)



class CheckInAPIView(APIView):
    authentication_classes = []
    permission_classes = []
    def post(self, request, *args, **kwargs):
        try:
            try:
                abs_path = handle_uploaded_image(request, 'temp_checkin.jpg')
            except ValueError as e:
                return Response({'error': str(e)}, status=http_status.HTTP_400_BAD_REQUEST)
            try:
                results = recognize_and_cleanup(abs_path)
            except Exception:
                return Response({'error': 'Error in face detection/recognition.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            response_data = []
            current_datetime = timezone.localtime(timezone.now())
            today = current_datetime.date()
            current_time = current_datetime.time()
            for face in results:
                try:
                    name = face['name']
                    if name.lower() == 'unknown':
                        response_data.append(build_student_data(face, None, 'unknown', None, None))
                        continue
                    student = Student.objects.filter(name=name).first()
                    attendance_status = 'unknown'
                    attendance_id = None
                    attendance_details = None
                    if student:
                        existing_attendance = Attendance.objects.filter(
                            student=student, date=today, course=None
                        ).first()
                        if existing_attendance:
                            if existing_attendance.check_in_time:
                                attendance_status = 'already_checked_in'
                                attendance_id = existing_attendance.id
                                attendance_details = {
                                    'status': existing_attendance.status,
                                    'check_in_time': existing_attendance.check_in_time.strftime('%H:%M'),
                                    'check_out_time': existing_attendance.check_out_time.strftime('%H:%M') if existing_attendance.check_out_time else None
                                }
                            else:
                                attendance_status = 'error'
                        else:
                            from .models import AttendanceSettings
                            settings = AttendanceSettings.get_settings()
                            if current_time <= settings.on_time_threshold:
                                attendance_status_value = 'present'
                            elif current_time <= settings.late_threshold:
                                attendance_status_value = 'late'
                            else:
                                attendance_status_value = 'absent'
                            att = Attendance.objects.create(
                                student=student,
                                date=today,
                                course=None,
                                check_in_time=current_time,
                                status=attendance_status_value,
                                confidence=1.0
                            )
                            attendance_status = 'checked_in'
                            attendance_id = att.id
                            attendance_details = {
                                'status': attendance_status_value,
                                'check_in_time': current_time.strftime('%H:%M'),
                                'check_out_time': None
                            }
                    response_data.append(build_student_data(face, student, attendance_status, attendance_id, attendance_details))
                except Exception:
                    continue
            return Response({'results': response_data}, status=http_status.HTTP_200_OK)
        except Exception:
            return Response({'error': 'Internal server error.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


class CheckOutAPIView(APIView):
    authentication_classes = []
    permission_classes = []
    def post(self, request, *args, **kwargs):
        try:
            try:
                abs_path = handle_uploaded_image(request, 'temp_checkout.jpg')
            except ValueError as e:
                return Response({'error': str(e)}, status=http_status.HTTP_400_BAD_REQUEST)
            try:
                results = recognize_and_cleanup(abs_path)
            except Exception:
                return Response({'error': 'Error in face detection/recognition.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            response_data = []
            current_datetime = timezone.localtime(timezone.now())
            today = current_datetime.date()
            current_time = current_datetime.time()
            for face in results:
                try:
                    name = face['name']
                    if name.lower() == 'unknown':
                        response_data.append(build_student_data(face, None, 'unknown', None, None))
                        continue
                    student = Student.objects.filter(name=name).first()
                    attendance_status = 'unknown'
                    attendance_id = None
                    attendance_details = None
                    if student:
                        existing_attendance = Attendance.objects.filter(
                            student=student, date=today, course=None
                        ).first()
                        if existing_attendance:
                            if existing_attendance.check_out_time:
                                attendance_status = 'already_checked_out'
                                attendance_id = existing_attendance.id
                                attendance_details = {
                                    'status': existing_attendance.status,
                                    'check_in_time': existing_attendance.check_in_time.strftime('%H:%M') if existing_attendance.check_in_time else None,
                                    'check_out_time': existing_attendance.check_out_time.strftime('%H:%M')
                                }
                            elif existing_attendance.check_in_time:
                                existing_attendance.check_out_time = current_time
                                existing_attendance.save()
                                attendance_status = 'checked_out'
                                attendance_id = existing_attendance.id
                                attendance_details = {
                                    'status': existing_attendance.status,
                                    'check_in_time': existing_attendance.check_in_time.strftime('%H:%M'),
                                    'check_out_time': current_time.strftime('%H:%M')
                                }
                            else:
                                attendance_status = 'error'
                        else:
                            attendance_status = 'no_check_in'
                    response_data.append(build_student_data(face, student, attendance_status, attendance_id, attendance_details))
                except Exception:
                    continue
            return Response({'results': response_data}, status=http_status.HTTP_200_OK)
        except Exception:
            return Response({'error': 'Internal server error.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


class RetrainModelAPIView(APIView):
    """API endpoint to manually trigger model retraining"""
    
    def post(self, request, *args, **kwargs):
        """Trigger manual model retraining"""
        try:
            from django.core.management import call_command
            import threading
            
            def retrain_async():
                try:
                    call_command('retrain_facenet', '--force', verbosity=1)
                    from .utils.face_pipeline import reload_models
                    reload_models()
                except Exception as e:
                    logger.error(f"Error in manual retraining: {str(e)}")
            
            thread = threading.Thread(target=retrain_async)
            thread.daemon = True
            thread.start()
            
            return Response({
                'message': 'Model retraining started in background',
                'status': 'processing'
            }, status=http_status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error triggering manual retraining: {str(e)}")
            return Response({
                'error': 'Failed to start retraining'
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)

@login_required
@user_passes_test(is_admin)
def verify_students(request):
    if request.method == 'POST':
        student_id = request.POST.get('student_id')
        action = request.POST.get('action')
        student = get_object_or_404(Student, id=student_id)
        if action == 'verify':
            roll_number = request.POST.get('roll_number', '').strip()
            # Require roll number if not already set
            if not student.roll_number and not roll_number:
                messages.error(request, f"Please provide a roll number for {student.name} before verifying.")
                return redirect('verify_students')
            # If provided, validate uniqueness and assign
            if roll_number and roll_number != student.roll_number:
                if Student.objects.filter(roll_number=roll_number).exclude(id=student.id).exists():
                    messages.error(request, f"Roll number {roll_number} is already in use.")
                    return redirect('verify_students')
                student.roll_number = roll_number
                # Optionally align auth username with roll number
                if student.user:
                    student.user.username = roll_number
                    student.user.save()
            student.is_verified = True
            student.save()
            messages.success(request, f"Student {student.name} has been verified.")
        elif action == 'reject':
            student_name = student.name
            if student.user:
                student.user.delete()
            student.delete()
            messages.success(request, f"Student {student_name} has been rejected and deleted.")
        return redirect('verify_students')
    unverified_students = Student.objects.filter(is_verified=False)
    context = {'unverified_students': unverified_students, 'unverified_count': get_unverified_count(request)}
    return render(request, 'teacherDashboard/verify_students.html', context)


@login_required
@user_passes_test(is_admin)
def performance_prediction(request):
    """Performance prediction page for students"""
    students = Student.objects.filter(is_verified=True).order_by('name')
    predictions = StudentPerformance.objects.select_related('student').order_by('-created_at')
    
    # Handle search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        students = students.filter(name__icontains=search_query)
        predictions = predictions.filter(student__name__icontains=search_query)
    
    # Handle form submission for new prediction
    if request.method == 'POST':
        try:
            student_id = request.POST.get('student_id')
            student = get_object_or_404(Student, id=student_id)
            
            # Get form data
            attendance_score = float(request.POST.get('attendance_score', 0))
            previous_grades = float(request.POST.get('previous_grades', 0))
            assignment_completion = float(request.POST.get('assignment_completion', 0))
            class_activeness = float(request.POST.get('class_activeness', 0))
            
            # Get weights (optional, use defaults if not provided)
            attendance_weight = float(request.POST.get('attendance_weight', 30.0))
            grades_weight = float(request.POST.get('grades_weight', 40.0))
            assignment_weight = float(request.POST.get('assignment_weight', 20.0))
            activeness_weight = float(request.POST.get('activeness_weight', 10.0))
            
            # Validate weights sum to 100%
            total_weight = attendance_weight + grades_weight + assignment_weight + activeness_weight
            if abs(total_weight - 100.0) > 0.1:  # Allow small floating point differences
                messages.error(request, f'Weights must sum to 100%. Current sum: {total_weight:.1f}%')
                return redirect('performance_prediction')
            
            # Validate score ranges
            scores = [attendance_score, previous_grades, assignment_completion, class_activeness]
            weights = [attendance_weight, grades_weight, assignment_weight, activeness_weight]
            
            for score in scores:
                if not (0 <= score <= 100):
                    messages.error(request, 'All scores must be between 0 and 100.')
                    return redirect('performance_prediction')
            
            for weight in weights:
                if weight < 0:
                    messages.error(request, 'All weights must be positive numbers.')
                    return redirect('performance_prediction')
            
            # Create or update performance prediction
            performance, created = StudentPerformance.objects.update_or_create(
                student=student,
                defaults={
                    'attendance_score': attendance_score,
                    'previous_grades': previous_grades,
                    'assignment_completion': assignment_completion,
                    'class_activeness': class_activeness,
                    'attendance_weight': attendance_weight,
                    'grades_weight': grades_weight,
                    'assignment_weight': assignment_weight,
                    'activeness_weight': activeness_weight,
                    'predicted_by': request.user,
                }
            )
            
            action = "created" if created else "updated"
            messages.success(request, 
                f'Performance prediction {action} for {student.name}. '
                f'Overall Score: {performance.overall_score:.1f}% '
                f'({performance.get_performance_category_display()})')
            
        except ValueError as e:
            messages.error(request, 'Invalid input values. Please enter valid numbers.')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
        
        return redirect('performance_prediction')
    
    # Paginate predictions
    paginator = Paginator(predictions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'students': students,
        'predictions': page_obj,
        'search_query': search_query,
        'unverified_count': get_unverified_count(request),
    }
    
    return render(request, 'teacherDashboard/performance_prediction.html', context)


@csrf_exempt
def performance_prediction_api(request, student_id):
    """API endpoint to get/update student performance prediction"""
    student = get_object_or_404(Student, id=student_id)
    
    if request.method == 'GET':
        # Get existing prediction or return default values
        try:
            performance = StudentPerformance.objects.get(student=student)
            data = {
                'success': True,
                'student_name': student.name,
                'attendance_score': performance.attendance_score,
                'previous_grades': performance.previous_grades,
                'assignment_completion': performance.assignment_completion,
                'class_activeness': performance.class_activeness,
                'attendance_weight': performance.attendance_weight,
                'grades_weight': performance.grades_weight,
                'assignment_weight': performance.assignment_weight,
                'activeness_weight': performance.activeness_weight,
                'overall_score': performance.overall_score,
                'performance_category': performance.get_performance_category_display(),
            }
        except StudentPerformance.DoesNotExist:
            data = {
                'success': True,
                'student_name': student.name,
                'attendance_score': 0,
                'previous_grades': 0,
                'assignment_completion': 0,
                'class_activeness': 0,
                'attendance_weight': 30.0,
                'grades_weight': 40.0,
                'assignment_weight': 20.0,
                'activeness_weight': 10.0,
                'overall_score': 0,
                'performance_category': 'Not predicted yet',
            }
        
        return JsonResponse(data)
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})