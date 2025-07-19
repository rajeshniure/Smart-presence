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
from .models import Student, Attendance, EmotionLog
import base64
import io
import json
from PIL import Image
from datetime import date, datetime, timedelta
import calendar

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status as http_status
from django.utils import timezone
from .models import Attendance, Student, Course
from .serializers import AttendanceSerializer
from .utils.face_pipeline import detect_and_recognize, test_pipeline
import os
from django.core.files.storage import default_storage
import logging
from django.utils.decorators import method_decorator

# Set up logging
logger = logging.getLogger(__name__)

# Helper functions
def is_admin(user):
    return user.is_staff or user.is_superuser

def is_student(user):
    return hasattr(user, 'student_profile') and user.student_profile is not None

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
        roll_number = request.POST.get('roll_number')
        password = request.POST.get('password')
        
        try:
            student = Student.objects.get(roll_number=roll_number)
            if not student.user:
                messages.error(request, "This student is not registered for login. Please contact your teacher.")
                return redirect('landing')
            user = authenticate(request, username=student.user.username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {student.name}!")
                return redirect('student_home')
            messages.error(request, "Invalid credentials.")
        except Student.DoesNotExist:
            messages.error(request, "Student with this roll number does not exist.")
        
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
    return render(request, "teacherDashboard/home.html", context)

@login_required
@user_passes_test(is_admin)
def register(request):
    if request.method == 'POST':
        try:
            # Extract form data
            name = request.POST.get('name')
            roll_number = request.POST.get('roll_number')
            email = request.POST.get('email')
            phone = request.POST.get('phone', '')
            department = request.POST.get('department')
            password = request.POST.get('password', '')
            confirm_password = request.POST.get('confirm_password', '')
            
            # Enforce password requirement
            if not password:
                messages.error(request, 'Password is required for student login access.')
                return render(request, "teacherDashboard/register.html")
            # Confirm password match
            if password != confirm_password:
                messages.error(request, 'Passwords do not match.')
                return render(request, "teacherDashboard/register.html")
            
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
                return render(request, "teacherDashboard/register.html")
            
            # Create user account
            username = roll_number  # Use roll number as username
            user = User.objects.create_user(username=username, email=email, password=password)
            user.first_name = name.split()[0] if ' ' in name else name
            user.last_name = name.split(' ', 1)[1] if ' ' in name else ''
            user.save()
            student.user = user
            
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
                messages.error(request, f'An error occurred during registration: {str(e)}')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
    
    return render(request, "teacherDashboard/register.html")

def scan(request):
    return render(request, "scan.html")

@login_required
@user_passes_test(is_admin)
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
    return render(request, "teacherDashboard/students.html", context)

@login_required
@user_passes_test(is_admin)
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
    total_days = Attendance.objects.filter(student=student).count()
    present_days = Attendance.objects.filter(student=student, status='present').count()
    late_days = Attendance.objects.filter(student=student, status='late').count()
    absent_days = Attendance.objects.filter(student=student, status='absent').count()
    
    attendance_rate = 0
    if total_days > 0:
        # Count late as half present for rate calculation
        attendance_rate = round(((present_days + (late_days * 0.5)) / total_days) * 100)
    
    attendance_stats = {
        'present': present_days,
        'late': late_days,
        'absent': absent_days,
        'rate': attendance_rate
    }
    
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
    total_records = attendance_records.count()
    present_count = attendance_records.filter(status='present').count()
    late_count = attendance_records.filter(status='late').count()
    absent_count = attendance_records.filter(status='absent').count()
    
    attendance_rate = 0
    if total_records > 0:
        # Count late as half present for rate calculation
        attendance_rate = round(((present_count + (late_count * 0.5)) / total_records) * 100)
    
    # Prepare month choices for the filter
    months = [(str(i), calendar.month_name[i]) for i in range(1, 13)]
    
    # Paginate the results
    paginator = Paginator(attendance_records, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'student': student,
        'attendance_records': page_obj,
        'attendance_stats': {
            'present': present_count,
            'late': late_count,
            'absent': absent_count,
            'rate': attendance_rate
        },
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
        
        if image_file:
            student.image = image_file
        elif webcam_image:
            # Process base64 webcam image
            format, imgstr = webcam_image.split(';base64,')
            ext = format.split('/')[-1]
            image_data = ContentFile(base64.b64decode(imgstr), name=f'{student.roll_number}_webcam.{ext}')
            student.image = image_data
        else:
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




def scan_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        temp_path = default_storage.save('temp_scan.jpg', image_file)
        abs_path = default_storage.path(temp_path)
        results = detect_and_recognize(abs_path)
        # Optionally, delete temp file after processing
        # default_storage.delete(temp_path)
        return render(request, 'scan.html', {'results': results})
    return render(request, 'scan.html')

class ScanAPIView(APIView):
    authentication_classes = []  # Disable authentication
    permission_classes = []      # Disable permissions
    
    def post(self, request, *args, **kwargs):
        try:
            logger.info("Scan API called")
            
            # Check if image file is provided
            image_file = request.FILES.get('image')
            if not image_file:
                logger.error("No image file provided in request")
                return Response({'error': 'No image provided.'}, status=http_status.HTTP_400_BAD_REQUEST)
            
            logger.info(f"Received image: {image_file.name}, size: {image_file.size}")
            
            # Save image temporarily
            try:
                temp_path = default_storage.save('temp_scan.jpg', image_file)
                abs_path = default_storage.path(temp_path)
                logger.info(f"Image saved to: {abs_path}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                return Response({'error': 'Error processing image.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Run face detection and recognition
            try:
                logger.info("Starting face detection and recognition...")
                results = detect_and_recognize(abs_path)
                logger.info(f"Detection/recognition results: {results}")
            except Exception as e:
                logger.error(f"Error in face detection/recognition: {str(e)}")
                return Response({'error': 'Error in face detection/recognition.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                # Clean up temp file
                try:
                    os.remove(abs_path)
                    logger.info("Temporary image file cleaned up")
                except:
                    pass
            
            # Process results and mark attendance
            response_data = []
            current_datetime = timezone.localtime(timezone.now())
            today = current_datetime.date()
            
            for face in results:
                try:
                    name = face['name']
                    box = face['box']
                    logger.info(f"Processing face: {name}, box: {box}")
                    
                    # Find student
                    student = Student.objects.filter(name=name).first()
                    attendance_status = 'unknown'
                    attendance_id = None
                    attendance_details = None
                    
                    if student:
                        logger.info(f"Student found: {student.name}")
                        current_datetime = timezone.localtime(timezone.now())
                        current_time = current_datetime.time()
                        current_date = current_datetime.date()
                        
                        # Check if attendance already exists for today
                        existing_attendance = Attendance.objects.filter(
                            student=student,
                            date=current_date,
                            course=None
                        ).first()
                        
                        if existing_attendance:
                            # Attendance already marked today
                            attendance_status = 'already_marked'
                            attendance_id = existing_attendance.id
                            attendance_details = {
                                'status': existing_attendance.status,
                                'check_in_time': existing_attendance.check_in_time.strftime('%H:%M') if existing_attendance.check_in_time else None,
                                'check_out_time': existing_attendance.check_out_time.strftime('%H:%M') if existing_attendance.check_out_time else None
                            }
                            logger.info(f"Attendance already marked: {existing_attendance.status}")
                        else:
                            # Mark new attendance with time-based logic
                            from .models import AttendanceSettings
                            settings = AttendanceSettings.get_settings()
                            
                            # Determine attendance status based on current time
                            if current_time <= settings.on_time_threshold:
                                attendance_status_value = 'present'
                                logger.info(f"Marking as present (time: {current_time}, threshold: {settings.on_time_threshold})")
                            elif current_time <= settings.late_threshold:
                                attendance_status_value = 'late'
                                logger.info(f"Marking as late (time: {current_time}, threshold: {settings.late_threshold})")
                            else:
                                attendance_status_value = 'absent'
                                logger.info(f"Marking as absent (time: {current_time}, threshold: {settings.absent_threshold})")
                            
                            # Create new attendance record
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
                            logger.info(f"New attendance marked: {attendance_status_value}")
                    else:
                        logger.warning(f"Student not found for name: {name}")
                    
                    # Prepare student data
                    student_data = {
                        'name': name,
                        'box': box,
                        'attendance_status': attendance_status,
                        'attendance_id': attendance_id,
                        'attendance_details': attendance_details,
                    }
                    
                    # Add student details if found
                    if student:
                        student_data.update({
                            'roll_number': student.roll_number,
                            'department': student.department,
                            'photo_url': student.image.url if student.image else None,
                        })
                    else:
                        # For unknown faces, still provide structure
                        student_data.update({
                            'roll_number': None,
                            'department': None,
                            'photo_url': None,
                        })
                    
                    response_data.append(student_data)
                    
                except Exception as e:
                    logger.error(f"Error processing face result: {str(e)}")
                    continue
            
            logger.info(f"Final response data: {response_data}")
            return Response({'results': response_data}, status=http_status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Unexpected error in ScanAPIView: {str(e)}")
            return Response({'error': 'Internal server error.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)

# Add a test endpoint for debugging
class TestPipelineAPIView(APIView):
    authentication_classes = []  # Disable authentication
    permission_classes = []      # Disable permissions
    
    def get(self, request, *args, **kwargs):
        """Test the face detection and recognition pipeline"""
        try:
            logger.info("Testing face detection and recognition pipeline...")
            success = test_pipeline()
            return Response({
                'success': success,
                'message': 'Pipeline test completed successfully' if success else 'Pipeline test failed'
            }, status=http_status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error testing pipeline: {str(e)}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


class CheckInAPIView(APIView):
    authentication_classes = []  # Disable authentication
    permission_classes = []      # Disable permissions
    
    def post(self, request, *args, **kwargs):
        """API endpoint for manual check-in"""
        try:
            logger.info("Check-in API called")
            
            # Check if image file is provided
            image_file = request.FILES.get('image')
            if not image_file:
                logger.error("No image file provided in request")
                return Response({'error': 'No image provided.'}, status=http_status.HTTP_400_BAD_REQUEST)
            
            logger.info(f"Received image: {image_file.name}, size: {image_file.size}")
            
            # Save image temporarily
            try:
                temp_path = default_storage.save('temp_checkin.jpg', image_file)
                abs_path = default_storage.path(temp_path)
                logger.info(f"Image saved to: {abs_path}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                return Response({'error': 'Error processing image.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Run face detection and recognition
            try:
                logger.info("Starting face detection and recognition for check-in...")
                results = detect_and_recognize(abs_path)
                logger.info(f"Detection/recognition results: {results}")
            except Exception as e:
                logger.error(f"Error in face detection/recognition: {str(e)}")
                return Response({'error': 'Error in face detection/recognition.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                # Clean up temp file
                try:
                    os.remove(abs_path)
                    logger.info("Temporary image file cleaned up")
                except:
                    pass
            
            # Process results and mark check-in
            response_data = []
            current_datetime = timezone.localtime(timezone.now())
            today = current_datetime.date()
            current_time = current_datetime.time()
            
            for face in results:
                try:
                    name = face['name']
                    box = face['box']
                    logger.info(f"Processing face for check-in: {name}, box: {box}")
                    
                    # Find student
                    student = Student.objects.filter(name=name).first()
                    attendance_status = 'unknown'
                    attendance_id = None
                    attendance_details = None
                    
                    if student:
                        logger.info(f"Student found for check-in: {student.name}")
                        
                        # Check if attendance already exists for today
                        existing_attendance = Attendance.objects.filter(
                            student=student,
                            date=today,
                            course=None
                        ).first()
                        
                        if existing_attendance:
                            if existing_attendance.check_in_time:
                                # Already checked in today
                                attendance_status = 'already_checked_in'
                                attendance_id = existing_attendance.id
                                attendance_details = {
                                    'status': existing_attendance.status,
                                    'check_in_time': existing_attendance.check_in_time.strftime('%H:%M'),
                                    'check_out_time': existing_attendance.check_out_time.strftime('%H:%M') if existing_attendance.check_out_time else None
                                }
                                logger.info(f"Student already checked in: {existing_attendance.check_in_time}")
                            else:
                                # Has attendance record but no check-in time (shouldn't happen normally)
                                attendance_status = 'error'
                                logger.warning(f"Student has attendance record but no check-in time")
                        else:
                            # Create new attendance record for check-in
                            from .models import AttendanceSettings
                            settings = AttendanceSettings.get_settings()
                            
                            # Determine attendance status based on current time
                            if current_time <= settings.on_time_threshold:
                                attendance_status_value = 'present'
                                logger.info(f"Check-in as present (time: {current_time}, threshold: {settings.on_time_threshold})")
                            elif current_time <= settings.late_threshold:
                                attendance_status_value = 'late'
                                logger.info(f"Check-in as late (time: {current_time}, threshold: {settings.late_threshold})")
                            else:
                                attendance_status_value = 'absent'
                                logger.info(f"Check-in as absent (time: {current_time}, threshold: {settings.absent_threshold})")
                            
                            # Create new attendance record
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
                            logger.info(f"New check-in marked: {attendance_status_value}")
                    else:
                        logger.warning(f"Student not found for check-in: {name}")
                    
                    # Prepare student data
                    student_data = {
                        'name': name,
                        'box': box,
                        'attendance_status': attendance_status,
                        'attendance_id': attendance_id,
                        'attendance_details': attendance_details,
                    }
                    
                    # Add student details if found
                    if student:
                        student_data.update({
                            'roll_number': student.roll_number,
                            'department': student.department,
                            'photo_url': student.image.url if student.image else None,
                        })
                    else:
                        # For unknown faces, still provide structure
                        student_data.update({
                            'roll_number': None,
                            'department': None,
                            'photo_url': None,
                        })
                    
                    response_data.append(student_data)
                    
                except Exception as e:
                    logger.error(f"Error processing face result for check-in: {str(e)}")
                    continue
            
            logger.info(f"Final check-in response data: {response_data}")
            return Response({'results': response_data}, status=http_status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Unexpected error in CheckInAPIView: {str(e)}")
            return Response({'error': 'Internal server error.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


class CheckOutAPIView(APIView):
    authentication_classes = []  # Disable authentication
    permission_classes = []      # Disable permissions
    
    def post(self, request, *args, **kwargs):
        """API endpoint for manual check-out"""
        try:
            logger.info("Check-out API called")
            
            # Check if image file is provided
            image_file = request.FILES.get('image')
            if not image_file:
                logger.error("No image file provided in request")
                return Response({'error': 'No image provided.'}, status=http_status.HTTP_400_BAD_REQUEST)
            
            logger.info(f"Received image: {image_file.name}, size: {image_file.size}")
            
            # Save image temporarily
            try:
                temp_path = default_storage.save('temp_checkout.jpg', image_file)
                abs_path = default_storage.path(temp_path)
                logger.info(f"Image saved to: {abs_path}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                return Response({'error': 'Error processing image.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Run face detection and recognition
            try:
                logger.info("Starting face detection and recognition for check-out...")
                results = detect_and_recognize(abs_path)
                logger.info(f"Detection/recognition results: {results}")
            except Exception as e:
                logger.error(f"Error in face detection/recognition: {str(e)}")
                return Response({'error': 'Error in face detection/recognition.'}, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                # Clean up temp file
                try:
                    os.remove(abs_path)
                    logger.info("Temporary image file cleaned up")
                except:
                    pass
            
            # Process results and mark check-out
            response_data = []
            current_datetime = timezone.localtime(timezone.now())
            today = current_datetime.date()
            current_time = current_datetime.time()
            
            for face in results:
                try:
                    name = face['name']
                    box = face['box']
                    logger.info(f"Processing face for check-out: {name}, box: {box}")
                    
                    # Find student
                    student = Student.objects.filter(name=name).first()
                    attendance_status = 'unknown'
                    attendance_id = None
                    attendance_details = None
                    
                    if student:
                        logger.info(f"Student found for check-out: {student.name}")
                        
                        # Check if attendance exists for today
                        existing_attendance = Attendance.objects.filter(
                            student=student,
                            date=today,
                            course=None
                        ).first()
                        
                        if existing_attendance:
                            if existing_attendance.check_out_time:
                                # Already checked out today
                                attendance_status = 'already_checked_out'
                                attendance_id = existing_attendance.id
                                attendance_details = {
                                    'status': existing_attendance.status,
                                    'check_in_time': existing_attendance.check_in_time.strftime('%H:%M') if existing_attendance.check_in_time else None,
                                    'check_out_time': existing_attendance.check_out_time.strftime('%H:%M')
                                }
                                logger.info(f"Student already checked out: {existing_attendance.check_out_time}")
                            elif existing_attendance.check_in_time:
                                # Has check-in but no check-out - mark check-out
                                existing_attendance.check_out_time = current_time
                                existing_attendance.save()
                                
                                attendance_status = 'checked_out'
                                attendance_id = existing_attendance.id
                                attendance_details = {
                                    'status': existing_attendance.status,
                                    'check_in_time': existing_attendance.check_in_time.strftime('%H:%M'),
                                    'check_out_time': current_time.strftime('%H:%M')
                                }
                                logger.info(f"Check-out marked: {current_time}")
                            else:
                                # Has attendance record but no check-in time (shouldn't happen normally)
                                attendance_status = 'error'
                                logger.warning(f"Student has attendance record but no check-in time")
                        else:
                            # No attendance record for today
                            attendance_status = 'no_check_in'
                            logger.warning(f"No check-in record found for student: {student.name}")
                    
                    # Prepare student data
                    student_data = {
                        'name': name,
                        'box': box,
                        'attendance_status': attendance_status,
                        'attendance_id': attendance_id,
                        'attendance_details': attendance_details,
                    }
                    
                    # Add student details if found
                    if student:
                        student_data.update({
                            'roll_number': student.roll_number,
                            'department': student.department,
                            'photo_url': student.image.url if student.image else None,
                        })
                    else:
                        # For unknown faces, still provide structure
                        student_data.update({
                            'roll_number': None,
                            'department': None,
                            'photo_url': None,
                        })
                    
                    response_data.append(student_data)
                    
                except Exception as e:
                    logger.error(f"Error processing face result for check-out: {str(e)}")
                    continue
            
            logger.info(f"Final check-out response data: {response_data}")
            return Response({'results': response_data}, status=http_status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Unexpected error in CheckOutAPIView: {str(e)}")
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
                    # Reload models after retraining
                    from .utils.face_pipeline import reload_models
                    reload_models()
                except Exception as e:
                    logger.error(f"Error in manual retraining: {str(e)}")
            
            # Start retraining in background thread
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
