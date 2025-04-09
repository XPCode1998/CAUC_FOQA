# views.py
from django.shortcuts import render
from ..models import QAR
from django.db.models import Max, Min, Avg

def index(request):
    
    return render(request, 'index.html')

