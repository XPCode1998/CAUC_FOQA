from django.shortcuts import render

# 首页
def index(request):
    return render(request, 'index.html')
