from . import views
from apps.qar_data import views as qar_views
from django.urls import path

urlpatterns = [
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
    path('', views.login, name='index'),
    path('qar_upload', qar_views.qar_upload, name='qar_upload'),
    path('qar_data_table', qar_views.qar_data, name='qar_data'),
]