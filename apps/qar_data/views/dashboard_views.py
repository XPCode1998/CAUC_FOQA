from django.shortcuts import render
from ..models import QAR
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from django.db.models import Avg, Max, Min, StdDev
from django.db import models
import json
from django.core.paginator import Paginator
from django.http import JsonResponse

# 飞行数据仪表盘
# 实时监控
# 快速告警
def monitor(request):
    # 获取所有QAR ID并按首字母排序
    qar_ids = QAR.objects.order_by('qar_id').values_list('qar_id', flat=True).distinct()
    # 获取第一个QAR ID
    first_qar_id = qar_ids[0]  # 直接获取第一个记录
    
    # 获取飞行数据
    flight_data = QAR.objects.filter(qar_id=first_qar_id)
    
    # 转换为DataFrame以便分析
    df = pd.DataFrame.from_records(flight_data.values())

    stats = {
        # 基础指标（已存在）
        'duration': round(df['dSimTime'].max() - df['dSimTime'].min(), 2),
        'fuel_consumed': round(df['gfuel'].iloc[0] - df['gfuel'].iloc[-1], 2),

        
        'max_altitude': round(df['dASL'].max(), 2),
        'min_altitude': round(df['dASL'].min(), 2),
        'avg_altitude': round(df['dASL'].mean(), 2),
        
        'max_speed': round(df['dTAS'].max(), 2),
        'min_speed': round(df['dTAS'].min(), 2),
        'avg_speed': round(df['dTAS'].mean(), 2),

        'max_vertical_speed': round(df['dWkg'].max(), 2),  # 最大垂直速度
        'min_vertical_speed': round(df['dWkg'].min(), 2),  # 最小垂直速度
        'avg_vertical_speed': round(df['dWkg'].mean(), 2),  # 平均垂直速度

        'max_mach': round(df['dMach'].max(), 2),
        'min_mach': round(df['dMach'].min(), 2),
        'avg_mach': round(df['dMach'].mean(), 2),

        'max_roll_angle': round(df['dPhi'].max(), 2),  # 最大滚转角
        'min_roll_angle': round(df['dPhi'].min(), 2),  # 最小滚转角
        'avg_roll_angle': round(df['dPhi'].mean(), 2),  # 平均滚转角

        'max_pitch_angle': round(df['dTheta'].max(), 2),  # 最大俯仰角
        'min_pitch_angle': round(df['dTheta'].min(), 2),  # 最小俯仰角
        'avg_pitch_angle': round(df['dTheta'].mean(), 2),  # 平均俯仰角
        
        'max_climb_rate': round(df['dGamma'].max(), 2),  # 最大爬升率
        'max_descent_rate': round(df['dGamma'].min(), 2),  # 最大下降率

        'max_g_force': round(max(df['dNx'].max(), df['dNy'].max(), df['dNz'].max()), 2),  # 最大过载
        'min_g_force': round(min(df['dNx'].min(), df['dNy'].min(), df['dNz'].min()), 2),  # 最小过载
        
    }

    time_label = df['dSimTime'].tolist()

    # 高度随时间变化
    altitude_line = df['dASL'].astype(int).tolist()  
    # 速度随时间变化
    speed_line =df['dTAS'].astype(int).tolist() 
    # 垂直速度随时间变化
    vertical_speed_line = df['dWkg'].astype(int).tolist()
    # 马赫数随时间变化
    mach_line = df['dMach'].astype(int).tolist()
    # 滚转角变化
    roll_angle_line = df['dPhi'].astype(int).tolist()
    # 俯仰角变化
    pitch_angle_line = df['dTheta'].astype(int).tolist()
    # 姿态角变化
    attitude_line = list(zip(df['dPhi'], df['dTheta'], df['dPsi']))
    # 过载变化
    g_force_line = list(zip(df['dNx'], df['dNy'], df['dNz']))

    context = {
        'qar_id': first_qar_id,
        'stats': stats,
        'altitude_line': altitude_line,
        'speed_line': speed_line,
        'vertical_speed_line': vertical_speed_line,
        'mach_line': mach_line,
        'roll_angle_line': roll_angle_line,
        'pitch_angle_line': pitch_angle_line,
        'g_force_line': g_force_line,
        'time_label': time_label,
    }

    
    return render(request, 'dashboard/monitor.html', context)




def incident(request):
    return render(request, 'dashboard/incident.html')



