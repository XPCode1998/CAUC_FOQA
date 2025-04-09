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
import json
import numpy as np



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
def qar_preview(request):
    # 获取字段列表作为表头（自动匹配模型字段）
    model_fields = [field.verbose_name for field in QAR._meta.fields]
    field_names = [field.name for field in QAR._meta.fields]

    # 获取页码参数，默认第一页
    page_number = request.GET.get('page', 1)
    per_page = 50  # 每页显示数量

    # 查询字段值（values生成dict更方便）
    query_set = QAR.objects.values(*field_names)
    paginator = Paginator(query_set, per_page)

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    # 将数据转换为 DataFrame
    df = pd.DataFrame(list(page_obj.object_list), columns=field_names)
    json_data = df.to_json(orient='values', date_format='iso')

    context = {
        'table_data': json_data,
        'table_headings': model_fields,
        'page_obj': page_obj
    }

    return render(request, 'data_manage/qar_preview.html', context)



# 数据质量监控
def qar_quality(request):
    return render(request, 'data_manage/qar_quality.html')


# 数据质量提升
def qar_imputation(request):
    # 获取字段列表作为表头
    model_fields = [field.verbose_name for field in QAR._meta.fields]
    field_names = [field.name for field in QAR._meta.fields]

    # 定义不允许掩码的字段
    exclude_fields = ['id', 'qar_id', 'flight_label', 'label','dSimTime', 'dStepTime']
    
    # 获取页码参数，默认第一页
    page_number = request.GET.get('page', 1)
    per_page = 50  # 每页显示数量

    # 查询字段值
    query_set = QAR.objects.values(*field_names)
    paginator = Paginator(query_set, per_page)

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(list(page_obj.object_list), columns=field_names)

    if request.method == 'GET':
       
        df_t = df.copy()
        
        # 1. 对允许掩码的字段进行处理
        mask_fields = [field for field in field_names if field not in exclude_fields]
        mask_values = ['', None, 'NULL', 'NA']    # 需要替换为NaN的值

        
        if not df_t.empty:
            mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
            df_t = df_t.mask(mask)
        
        for field in mask_fields:
            if field in df.columns:
                df[field] = df_t[field].replace(mask_values, pd.NA)
        
        # 2. 将DataFrame转换为前端需要的格式
        # 替换所有NA/NaN为字符串'NaN'，方便前端识别
        df = df.fillna('NaN')
    
    elif request.method == 'POST':
        pass
    
    # 转换为列表形式（而不是JSON），确保数据格式正确
    table_data = df.values.tolist()

    context = {
        'table_data': table_data,  # 直接传递列表数据
        'table_headings': model_fields,
        'page_obj': page_obj
    }
    return render(request, 'data_manage/qar_imputation.html', context)