from . import views
from django.urls import path

urlpatterns = [
    path('data_upload/', views.data_upload, name='data_upload'),
    path('data_preview/', views.data_preview, name='data_preview'),
    path('data_monitor/', views.data_monitor, name='data_monitor'),
    path('data_imputation/', views.data_imputation, name='data_imputation'),
]