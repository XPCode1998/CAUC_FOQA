from django.shortcuts import render

# 飞行数据仪表盘
# 实时监控
def incident(request):
    return render(request, 'dashboard/incident.html')

# 快速告警
def monitor(request):
    return render(request, 'dashboard/monitor.html')
