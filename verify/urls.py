from django.urls import path
from . import views

urlpatterns = [
    path('', views.verify_signature, name='verify_signature'),
]
