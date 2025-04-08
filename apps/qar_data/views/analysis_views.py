from django.shortcuts import render

# 飞行事件分析
# 预设事件管理
def incident_config(request):
    return render(request, 'analysis/incident_config.html')

# 飞行事件监控
def incident_monitor(request):
    return render(request, 'analysis/incident_monitor.html')