from . import views
from django.urls import path

urlpatterns = [
    path('',views.flight_preview, name='index'),
    path('flight_preview/', views.flight_preview, name='flight_preview'),
    path('preset_parameter/', views.preset_parameter, name='preset_parameter'),
    path('flight_risk/', views.flight_risk, name='flight_risk'),
]