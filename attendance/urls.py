from django.urls import path
from . import views
from .views import ScanAPIView, CheckInAPIView, CheckOutAPIView, RetrainModelAPIView, attendance_edit_api, attendance_export_api

urlpatterns = [
    # Landing page
    path("", views.landing, name="landing"),
    
    # Authentication routes
    path("login/admin/", views.admin_login_view, name="admin_login"),
    path("login/student/", views.student_login_view, name="student_login"),
    path("logout/", views.logout_view, name="logout"),
    
    # Teacher/Admin routes (renamed from admin/ to teacher/ to avoid conflict with Django admin)
    path("teacher/dashboard/", views.home, name="home"),
    path("teacher/verify/", views.verify_students, name="verify_students"),
    path("teacher/students/", views.students, name="students"),
    path("teacher/attendance/", views.attendance, name="attendance"),
    path("teacher/performance/", views.performance_prediction, name="performance_prediction"),
    
    # Student dashboard routes
    path("student/dashboard/", views.student_home, name="student_home"),
    path("student/attendance/", views.student_attendance, name="student_attendance"),
    path("student/profile/", views.student_profile, name="student_profile"),
    path("student/profile/update/", views.student_profile_update, name="student_profile_update"),
    path("student/profile/photo/update/", views.student_photo_update, name="student_photo_update"),
    
    # Universal scan page
    path("scan/", views.scan, name="scan"),
    
    # Universal registration route
    path("register/", views.register, name="register"),
    
    # API endpoints
    path("api/student/<int:student_id>/", views.student_detail_api, name="student_detail_api"),
    path("api/student/<int:student_id>/update/", views.student_update_api, name="student_update_api"),
    path("api/student/<int:student_id>/delete/", views.student_delete_api, name="student_delete_api"),
    path("api/attendance/<int:record_id>/emotions/", views.attendance_emotions_api, name="attendance_emotions_api"),
    path('api/scan/', ScanAPIView.as_view(), name='api_scan'),
    path('api/checkin/', CheckInAPIView.as_view(), name='api_checkin'),
    path('api/checkout/', CheckOutAPIView.as_view(), name='api_checkout'),
    path('api/retrain/', RetrainModelAPIView.as_view(), name='api_retrain'),
    path('api/performance/<int:student_id>/', views.performance_prediction_api, name='performance_prediction_api'),

    # Attendance export
    path('api/attendance/export/', attendance_export_api, name='attendance_export_api'),

    path('api/attendance/<int:record_id>/edit/', attendance_edit_api, name='attendance_edit_api'),
] 