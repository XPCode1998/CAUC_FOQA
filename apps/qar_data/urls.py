from . import views
from django.urls import path

urlpatterns = [
    path('qar_upload', views.qar_upload, name='qar_upload'),
    path('qar_data_table', views.qar_data, name='qar_data'),
]