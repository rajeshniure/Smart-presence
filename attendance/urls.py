from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register, name="register"),
    path("scan/", views.scan, name="scan"),
    path("students/", views.students, name="students"),
    path("attendance/", views.attendance, name="attendance"),
    
    # API endpoints
    path("api/student/<int:student_id>/", views.student_detail_api, name="student_detail_api"),
    path("api/student/<int:student_id>/delete/", views.student_delete_api, name="student_delete_api"),
    path("api/attendance/<int:record_id>/emotions/", views.attendance_emotions_api, name="attendance_emotions_api"),
] 