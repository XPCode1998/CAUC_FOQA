from . import views
from apps.qar_data import views as qar_views
from django.urls import path

urlpatterns = [
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
]