from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('/map/', views.map_view, name = 'map'),
    path('/analyze/', views.analyze, name = 'analyze'),
    path("status/<str:job_id>/", views.job_status, name="job_status"),
]
