import os
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.files.uploadedfile import InMemoryUploadedFile
from ..forms import QARUploadForm
from ..models import QAR
from ..serializers import QARSerializer
from rest_framework import viewsets
from rest_framework.filters import SearchFilter, OrderingFilter
from django_filters.rest_framework import DjangoFilterBackend
import pandas as pd
from django.core.paginator import Paginator



# 飞行数据管理
# 飞行数据导入
def get_model_fields():
    """获取QAR模型中需要导入的字段列表（排除元字段）"""
    exclude_fields = ['id', 'qar_id', 'flight_label', 'label']
    return [f.name for f in QAR._meta.get_fields() 
           if f.name not in exclude_fields and not f.auto_created]

def qar_upload(request):
    if request.method == 'POST':
        print(request.POST)   # 打印表单数据
        print(request.FILES)  # 打印上传的文件

        form = QARUploadForm(request.POST, request.FILES)
        if form.is_valid():
            qar_id = form.cleaned_data['qar_id']
            flight_label = form.cleaned_data['label']
            file = form.cleaned_data['file']
            
            try:
                # 读取并验证文件内容
                try:
                    content = file.read().decode('utf-8').splitlines()
                    file.seek(0)  # 重置文件指针
                except UnicodeDecodeError:
                    messages.error(request, '文件编码错误，请使用UTF-8编码的文本文件')
                    return redirect('qar_upload')
                
                if not content:
                    messages.error(request, '文件内容为空')
                    return redirect('qar_upload')
                
                # 获取字段名（第一行）
                headers = [h.strip() for h in content[0].split() if h.strip()]
                
                # 获取模型字段
                model_fields = get_model_fields()
                
                # 验证字段
                if len(headers) != len(model_fields):
                    messages.error(request, 
                                f'字段数量不匹配。需要{len(model_fields)}个字段，实际{len(headers)}个')
                    return redirect('qar_upload')
                
                missing_fields = set(model_fields) - set(headers)
                if missing_fields:
                    messages.error(request, 
                                f'缺少必要字段: {", ".join(missing_fields)}')
                    return redirect('qar_upload')
                
                # 处理数据行
                success_count = 0
                error_lines = []
                
                for line_num, line in enumerate(content[1:], start=2):
                    line = line.strip()
                    if not line:
                        continue
                        
                    values = line.split()
                    
                    if len(values) != len(headers):
                        error_lines.append(f"行{line_num}: 字段数量不匹配")
                        continue
                    
                    # 准备数据字典
                    data_dict = {
                        'qar_id': qar_id,
                        'label': flight_label  # 使用label字段而不是flight_label
                    }
                    
                    for header, value in zip(headers, values):
                        if header in model_fields:
                            try:
                                field = QAR._meta.get_field(header)
                                if value == 'None':
                                    data_dict[header] = None
                                elif field.get_internal_type() in ['FloatField', 'IntegerField']:
                                    data_dict[header] = float(value) if value else None
                                else:
                                    data_dict[header] = value
                            except (ValueError, TypeError):
                                error_lines.append(f"行{line_num} {header}: 值'{value}'无效")
                                data_dict[header] = None
                    
                    # 保存数据
                    try:
                        QAR.objects.create(**data_dict)
                        success_count += 1
                    except Exception as e:
                        error_lines.append(f"行{line_num}: 保存失败 - {str(e)}")
                
                # 显示结果
                if success_count > 0:
                    label_text = dict(QARUploadForm.LABEL_CHOICES).get(int(flight_label), '未知')
                    msg = f'成功导入{success_count}条记录 (QAR ID: {qar_id}, 标签: {label_text})'
                    messages.success(request, msg)
                
                if error_lines:
                    error_msg = f"发现{len(error_lines)}个错误:\n" + "\n".join(error_lines[:5])
                    if len(error_lines) > 5:
                        error_msg += f"\n...(只显示前5个错误)"
                    messages.warning(request, error_msg)
                
                return redirect('qar_upload')
            
            except Exception as e:
                messages.error(request, f'处理文件时出错: {str(e)}')
                return redirect('qar_upload')
        else:
            # 表单验证错误
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{form.fields[field].label}: {error}")
    else:
        form = QARUploadForm()
    
    return render(request, 'data_manage/qar_upload.html', {
        'form': form,
        'model_fields': get_model_fields()  # 可用于前端显示期望的字段
    })


# 飞行数据预览
class QARViewSet(viewsets.ModelViewSet):
    queryset = QAR.objects.all()
    serializer_class = QARSerializer
    
    # 添加过滤、搜索和排序功能
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    
    # 可过滤字段
    filterset_fields = {
        'qar_id': ['exact', 'contains'],
        'dSimTime': ['gte', 'lte', 'exact'],
        'dASL': ['gte', 'lte'],
        'dGroundspeed': ['gte', 'lte'],
        'label': ['exact'],
    }
    
    # 可搜索字段
    search_fields = ['qar_id']
    
    # 可排序字段
    ordering_fields = ['dSimTime', 'dASL', 'dGroundspeed']
    ordering = ['dSimTime']  # 默认排序


def qar_preview(request):
    # 定义表头（保持与原始格式一致）
    headings = [
        'QAR ID', '模拟时间', '单步运行模拟时间', '重力加速度', '风速沿地轴X轴的分量', '风速沿地轴Y轴的分量',
        '风速沿地轴Z轴的分量', '紊流沿机体轴X轴的分量', '紊流沿机体轴Y轴的分量', '紊流沿机体轴Z轴的分量',
        '飞机质量', '迎角', '迎角', '迎角正弦值', '迎角余弦值', '迎角变化率', '侧滑角', '侧滑角',
        '侧滑角正弦值', '侧滑角余弦值', '侧滑角变化率', '滚转角', '俯仰角', '偏航角', '航迹方位角',
        '航迹爬升角', '航迹速度', '真空速', '马赫数', '动压', '静压', '航迹速度沿机体轴X轴的分量',
        '航迹速度沿机体轴Y轴的分量', '航迹速度沿机体轴Z轴的分量', '线加速度沿机体轴X轴的分量',
        '线加速度沿机体轴Y轴的分量', '线加速度沿机体轴Z轴的分量', '空速沿机体轴X轴的分量',
        '空速沿机体轴Y轴的分量', '空速沿机体轴Z轴的分量', '机体轴X轴角速度', '机体轴Y轴角速度',
        '机体轴Z轴角速度', '机体轴X轴角速度', '机体轴Y轴角速度', '机体轴Z轴角速度',
        '机体轴X轴角加速度', '机体轴Y轴角加速度', '机体轴Z轴角加速度', '机体轴重心处X轴的过载',
        '机体轴重心处Y轴的过载', '机体轴重心处Z轴的过载', '线加速度沿机地轴X轴的分量',
        '线加速度沿机地轴Y轴的分量', '线加速度沿机地轴Z轴的分量', '航迹速度沿地轴系X轴速度',
        '航迹速度沿地轴系Y轴速度', '航迹速度沿地轴系Z轴速度', '经度', '纬度', '飞机重心海拔高度',
        '飞机重心离地高度', '真航向', '磁航向', '当前飞机质心在前一时刻地轴系的X轴坐标',
        '当前飞机质心在前一时刻地轴系的Y轴坐标', '当前飞机质心在前一时刻地轴系的Z轴坐标',
        '副翼偏度', '方向舵偏度', '升降舵偏度', '起落架位置', '襟翼偏度', '剩余油量', '发动机油门杆位置1',
        '发动机油门杆位置2', '发动机转速1', '发动机转速2', '发动机推力', '耗油量', '燃油消耗率',
        '燃油消耗量', '余油量', '横向驾驶员（滚转）操纵位置（横向杆位移）', '航向驾驶员（偏航）操纵位置（脚蹬位移）',
        '纵向驾驶员（俯仰）操纵位置（纵向杆位移）', '配平滚转操纵量', '配平俯仰操纵量', '标签',
    ]
    
    variances = [0.08, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 
                 0.0, 0.0, 0.0, 0.0, 0.04, 0.02, 0.08, 0.08, 0.01, 0.04, 0.04, 0.04, 0.02, 0.06, 0.04, 
                 0.0, 0.01, 0.0, 0.0, 0.0, 0.04, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.07, 0.01, 0.03, 0.03, 0.06, 0.06, 0.08, 0.08, 
                 0.02, 0.02, 0.06, 0.0, 0.01, 0.0, 0.0, 0.0, 0.07, 0.16, 0.16, 0.21, 0.2, 0.0, 0.03, 
                 0.0, 0.03, 0.07, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0]
    
    labels = headings
    
    # 获取页码参数，默认为1
    page_number = request.GET.get('page', 1)
    
    # 设置每页显示的数量（可根据实际需求调整）
    per_page = 50
    
    query_set = QAR.objects.values_list(
    'qar_id', 'dSimTime', 'dStepTime', 'dGravityAcc', 'dUwg', 'dVwg', 'dWwg',
    'dUTrub', 'dVTrub', 'dWTrub', 'dMass', 'dAlpha', 'dAlphaRad', 'dSinAlpha',
    'dCosAlpha', 'dAlphaDot', 'dBeta', 'dBetaRad', 'dSinBeta', 'dCosBeta',
    'dBetaDot', 'dPhi', 'dTheta', 'dPsi', 'dChi', 'dGamma', 'dGroundspeed',
    'dTAS', 'dMach', 'dPd', 'dPs', 'dUk', 'dVk', 'dWk', 'dUkDot', 'dVkDot',
    'dWkDot', 'dU', 'dV', 'dW', 'dP', 'dQ', 'dR', 'dPRad', 'dQRad', 'dRRad',
    'dRDot', 'dPDot', 'dQDot', 'dNx', 'dNy', 'dNz', 'dUkgDot', 'dVkgDot',
    'dWkgDot', 'dUkg', 'dVkg', 'dWkg', 'dLongitude', 'dLatitude', 'dASL',
    'dAGL', 'dTrueHeading', 'dMagHeading', 'dPosXg', 'dPosYg', 'dPosZg',
    'dtx', 'dty', 'dtz', 'LGPos', 'dFlap', 'gfuel', 'pe_t1', 'pe_t2', 'rot1',
    'rot2', 'thrust', 'gfused', 'dGtNormal', 'dGfNormal', 'dGFuel', 'dDa',
    'dDr', 'dDe', 'dDaTrim', 'dDeTrim', 'label'
    )
    
    # 创建分页器
    paginator = Paginator(query_set, per_page)
    
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        # 如果页码不是整数，返回第一页
        page_obj = paginator.page(1)
    except EmptyPage:
        # 如果页码超出范围，返回最后一页
        page_obj = paginator.page(paginator.num_pages)
    
    # 将当前页数据转换为DataFrame
    df = pd.DataFrame(list(page_obj.object_list), columns=labels)

    print(df)
    
    # 转换为JSON格式
    json_data = df.to_json(orient='values', date_format='iso')
    
    context = {
        'json_data': json_data,
        'headings': headings,
        'variances': variances,
        'labels': labels,
        'page_obj': page_obj  # 分页对象，用于模板中显示分页导航
    }
    
    return render(request, 'data_manage/qar_preview.html', context)


# 数据质量监控
def qar_quality(request):
    return render(request, 'data_manage/qar_quality.html')


# 数据质量提升
def qar_imputation(request):
    return render(request, 'data_manage/qar_imputation.html')