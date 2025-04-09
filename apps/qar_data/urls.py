from . import views
from django.urls import path

urlpatterns = [
    # 首页
    path('', views.monitor, name='index'),
    path('index', views.monitor, name='index'),
    # 飞行数据仪表盘
    path('incident_config', views.incident_config, name='incident_config'),
    path('incident_monitor', views.incident_monitor, name='incident_monitor'),
    # 飞行数据管理
    path('qar_upload', views.qar_upload, name='qar_upload'),
    path('qar_preview', views.qar_preview, name='qar_preview'),
    path('qar_quality', views.qar_quality, name='qar_quality'),
    path('qar_imputation', views.qar_imputation, name='qar_imputation'),
    # 飞行事件分析
    path('incident', views.incident, name='incident'),
    path('monitor', views.monitor, name='monitor'),
]