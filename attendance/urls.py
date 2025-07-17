from django.urls import path
from . import views

urlpatterns = [
    # Landing page
    path("", views.landing, name="landing"),
    
    # Authentication routes
    path("login/admin/", views.admin_login_view, name="admin_login"),
    path("login/student/", views.student_login_view, name="student_login"),
    path("logout/", views.logout_view, name="logout"),
    
    # Teacher/Admin routes (renamed from admin/ to teacher/ to avoid conflict with Django admin)
    path("teacher/dashboard/", views.home, name="home"),
    path("teacher/register/", views.register, name="register"),
    path("teacher/students/", views.students, name="students"),
    path("teacher/attendance/", views.attendance, name="attendance"),
    
    # Student dashboard routes
    path("student/dashboard/", views.student_home, name="student_home"),
    path("student/attendance/", views.student_attendance, name="student_attendance"),
    path("student/profile/", views.student_profile, name="student_profile"),
    path("student/profile/update/", views.student_profile_update, name="student_profile_update"),
    path("student/profile/photo/update/", views.student_photo_update, name="student_photo_update"),
    
    # Universal scan page
    path("scan/", views.scan, name="scan"),
    
    # API endpoints
    path("api/student/<int:student_id>/", views.student_detail_api, name="student_detail_api"),
    path("api/student/<int:student_id>/update/", views.student_update_api, name="student_update_api"),
    path("api/student/<int:student_id>/delete/", views.student_delete_api, name="student_delete_api"),
    path("api/attendance/<int:record_id>/emotions/", views.attendance_emotions_api, name="attendance_emotions_api"),
] 