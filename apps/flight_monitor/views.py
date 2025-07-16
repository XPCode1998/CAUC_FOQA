from django.shortcuts import render,redirect
from apps.core.models import QAR, QAR_Parameter_Attribute
from .models import Flight_Trajectory
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
import seaborn as sns
from django.db.models import Avg, Max, Min, StdDev
from django.db import models
import json
from django.core.paginator import Paginator
from django.http import JsonResponse

from django.views.generic import ListView, TemplateView
from django.urls import reverse_lazy
from django.contrib import messages


def generate_flight_trajectory_plot(df):
    """
    Generate a 3D flight trajectory plot from flight data DataFrame
    
    Args:
        df (pd.DataFrame): Flight data containing longitude, latitude, altitude
        qar_id (str): Flight identifier for title
        
    Returns:
        str: Base64 encoded PNG image of the plot
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color gradient based on altitude
    colors = plt.cm.viridis((df['dASL'] - df['dASL'].min()) / (df['dASL'].max() - df['dASL'].min()))
    
    # Plot the flight path with color gradient
    for i in range(len(df)-1):
        ax.plot(df['dLongitude'].iloc[i:i+2], 
                df['dLatitude'].iloc[i:i+2], 
                df['dASL'].iloc[i:i+2], 
                color=colors[i], linewidth=1.5, alpha=0.7)
    
    # Mark start and end points
    ax.scatter(df['dLongitude'].iloc[0], df['dLatitude'].iloc[0], df['dASL'].iloc[0], 
               c='green', s=150, marker='o', edgecolor='black', label='Start')
    ax.scatter(df['dLongitude'].iloc[-1], df['dLatitude'].iloc[-1], df['dASL'].iloc[-1], 
               c='red', s=150, marker='s', edgecolor='black', label='End')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(vmin=df['dASL'].min(), vmax=df['dASL'].max()))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Altitude (m)', fontsize=10)
    
    # Configure axis labels
    ax.set_xlabel('Longitude', fontsize=10, labelpad=10)
    ax.set_ylabel('Latitude', fontsize=10, labelpad=10)
    ax.set_zlabel('Altitude (m)', fontsize=10, labelpad=10)
    
    # Set title and legend
    # ax.set_title(f'3D Flight Trajectory - QAR ID: {qar_id}', fontsize=12, pad=20)
    # ax.legend(fontsize=9, bbox_to_anchor=(0.7, 0.9))
    
    # Adjust view angle
    ax.view_init(elev=25, azim=45)
    ax.dist = 10
    
    # Save plot to buffer and convert to base64
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return image_base64


def flight_preview(request):
    # 获取所有QAR ID并按首字母排序
    qar_ids = QAR.objects.order_by('qar_id').values_list('qar_id', flat=True).distinct()
    
    # 获取查询参数
    qar_id = request.GET.get('qar_id', '')
    qar_id_t = qar_id
    
    # 如果没有提供qar_id，使用第一个QAR ID
    if not qar_id and qar_ids:
        qar_id = qar_ids[0]
    
    # 获取飞行数据
    flight_data = QAR.objects.filter(qar_id=qar_id)
            
    # 如果没有数据，返回空结果
    if not flight_data.exists():
        context = {
            'qar_id': qar_id,
            'error': f"未找到QAR ID为 {qar_id} 的飞行数据",
            'qar_ids': qar_ids,
        }
        return render(request, 'flight_monitor/flight_preview.html', context)
    
    # 转换为DataFrame以便分析
    df = pd.DataFrame.from_records(flight_data.values())

    stats = {
        'duration': round(df['dSimTime'].max() - df['dSimTime'].min(), 2),
        'fuel_consumed': round(df['gfuel'].iloc[0] - df['gfuel'].iloc[-1], 2),
        'max_altitude': round(df['dASL'].max(), 2),
        'min_altitude': round(df['dASL'].min(), 2),
        'avg_altitude': round(df['dASL'].mean(), 2),
        'max_speed': round(df['dTAS'].max(), 2),
        'min_speed': round(df['dTAS'].min(), 2),
        'avg_speed': round(df['dTAS'].mean(), 2),
        'max_vertical_speed': round(df['dWkg'].max(), 2),
        'min_vertical_speed': round(df['dWkg'].min(), 2),
        'avg_vertical_speed': round(df['dWkg'].mean(), 2),
        'max_mach': round(df['dMach'].max(), 2),
        'min_mach': round(df['dMach'].min(), 2),
        'avg_mach': round(df['dMach'].mean(), 2),
        'max_roll_angle': round(df['dPhi'].max(), 2),
        'min_roll_angle': round(df['dPhi'].min(), 2),
        'avg_roll_angle': round(df['dPhi'].mean(), 2),
        'max_pitch_angle': round(df['dTheta'].max(), 2),
        'min_pitch_angle': round(df['dTheta'].min(), 2),
        'avg_pitch_angle': round(df['dTheta'].mean(), 2),
        'max_climb_rate': round(df['dGamma'].max(), 2),
        'max_descent_rate': round(df['dGamma'].min(), 2),
        'max_g_force': round(max(df['dNx'].max(), df['dNy'].max(), df['dNz'].max()), 2),
        'min_g_force': round(min(df['dNx'].min(), df['dNy'].min(), df['dNz'].min()), 2),
    }

    time_label = df['dSimTime'].tolist()
    altitude_line = df['dASL'].astype(int).tolist()  
    speed_line = df['dTAS'].astype(int).tolist() 
    vertical_speed_line = df['dWkg'].astype(int).tolist()
    mach_line = df['dMach'].astype(int).tolist()
    roll_angle_line = df['dPhi'].astype(int).tolist()
    pitch_angle_line = df['dTheta'].astype(int).tolist()
    attitude_line = list(zip(df['dPhi'], df['dTheta'], df['dPsi']))
    g_force_line = list(zip(df['dNx'], df['dNy'], df['dNz']))

    
    trajectory_3d_image = generate_flight_trajectory_plot(df)

    
    context = {
        'qar_id_t': qar_id_t,
        'qar_id': qar_id,
        'qar_ids': qar_ids,
        'stats': stats,
        'altitude_line': altitude_line,
        'speed_line': speed_line,
        'vertical_speed_line': vertical_speed_line,
        'mach_line': mach_line,
        'roll_angle_line': roll_angle_line,
        'pitch_angle_line': pitch_angle_line,
        'g_force_line': g_force_line,
        'time_label': time_label,
        'trajectory_3d_image': trajectory_3d_image,
    }
    
    return render(request, 'flight_monitor/flight_preview.html', context)


def preset_parameter(request):
    """
    显示和设置QAR参数阈值的主视图
    """
    # 处理筛选条件
    monitored_only = request.GET.get('monitored_only')
    
    # 获取参数列表
    parameters = QAR_Parameter_Attribute.objects.all()
    if monitored_only:
        parameters = parameters.filter(is_monitored=True)
    
    # 按参数名排序
    parameters = parameters.order_by('parameter_name')
    
    # 处理POST请求(保存设置)
    if request.method == 'POST':
        try:
            for param in parameters:
                # 获取表单数据
                warning_lower = request.POST.get(f'{param.parameter_name}_warning_lower')
                warning_upper = request.POST.get(f'{param.parameter_name}_warning_upper')
                critical_lower = request.POST.get(f'{param.parameter_name}_critical_lower')
                critical_upper = request.POST.get(f'{param.parameter_name}_critical_upper')
                
                # 更新阈值设置
                param.warning_lower = float(warning_lower) if warning_lower else None
                param.warning_upper = float(warning_upper) if warning_upper else None
                param.critical_lower = float(critical_lower) if critical_lower else None
                param.critical_upper = float(critical_upper) if critical_upper else None
                
                # 更新监控状态
                param.is_monitored = request.POST.get(f'{param.parameter_name}_is_monitored') == 'on'
                
                param.save()
            
            messages.success(request, '参数阈值设置已成功保存！')
            return redirect('preset_parameter')  # 重定向到当前视图以清除POST数据
        except Exception as e:
            messages.error(request, f'保存失败: {str(e)}')
    
    # 渲染模板
    context = {
        'parameters': parameters,
    }
    return render(request, 'flight_monitor/preset_parameter.html', context)


def flight_risk(request):
    # 获取所有QAR ID用于自动补全
    qar_ids = QAR.objects.values_list('qar_id', flat=True).distinct()
    
    # 获取查询参数
    qar_id = request.GET.get('qar_id')
    if qar_id:
        qar_id_t = qar_id
    else:
        qar_id_t = ''
    

    if not qar_id:
        # 如果没有提供qar_id，使用第一个QAR ID
        qar_ids = list(qar_ids)
        qar_id = qar_ids[0]
    
    # 初始化结果
    flight_data = None
    exceeded_records = []
    param_attrs = {p.parameter_name: p for p in QAR_Parameter_Attribute.objects.all()}
    
    if qar_id:
        # 获取指定QAR ID的完整飞行记录
        flight_data = QAR.objects.filter(qar_id=qar_id).first()
        
        if flight_data:
            # 分析每条记录的参数超限情况
            for field in QAR.get_fields():
                param_attr = param_attrs.get(field)
                if not param_attr:
                    continue
                    
                value = getattr(flight_data, field)
                if value is None:
                    continue
                    
                # 检查是否超限
                alerts = []
                severity = None
                
                if param_attr.critical_lower is not None and value < param_attr.critical_lower:
                    alerts.append('严重下限')
                    severity = '高'
                elif param_attr.critical_upper is not None and value > param_attr.critical_upper:
                    alerts.append('严重上限')
                    severity = '高'
                elif param_attr.warning_lower is not None and value < param_attr.warning_lower:
                    alerts.append('警告下限')
                    severity = '中'
                elif param_attr.warning_upper is not None and value > param_attr.warning_upper:
                    alerts.append('警告上限')
                    severity = '中'
                    
                if alerts:
                    exceeded_records.append({
                        'parameter': field,
                        'parameter_name': param_attr.description or field,
                        'value': value,
                        'unit': param_attr.unit or '',
                        'thresholds': f"{param_attr.warning_lower or param_attr.critical_lower} ~ {param_attr.warning_upper or param_attr.critical_upper}",
                        'exceed_type': ' / '.join(alerts),
                        'severity': severity,
                    })
    
    context = {
        'qar_ids': qar_ids,
        'qar_id_t': qar_id_t,
        'qar_id': qar_id,
        'flight_data': flight_data,
        'exceeded_records': exceeded_records,
    }
    
    return render(request, 'flight_monitor/flight_risk.html', context)