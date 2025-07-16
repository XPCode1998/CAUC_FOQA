from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.db import transaction
import pandas as pd
from django.core.paginator import Paginator
from .forms import DataUploadForm
from apps.core.models import QAR, QAR_Parameter_Attribute
from .models import QAR_Mask
from apps.core.signals import update_qar_parameter_stats
from apps.data_management.signals import update_qar_mask
from apps.flight_monitor.signals import update_flight_trajectory
from .utils import calculate_flight_status_stats, calculate_field_missing_stats, update_label_missing_stats
from django.core.paginator import Paginator
import numpy as np
from apps.ml_models.LGTDM.main import run_lgtdm, LGTDMConfig
from apps.ml_models.LGTDM.model import LGTDM
from apps.ml_models.LGTDM.utils.tools import get_data_info
import torch


import io
import pandas as pd
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render

def data_upload(request):
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            qar_id = form.cleaned_data['qar_id']
            flight_label = form.cleaned_data['label']
            uploaded_file = form.cleaned_data['file']
            
            try:
                # 重置文件指针并读取内容
                uploaded_file.seek(0)
                file_content = uploaded_file.read()
                
                # 尝试解码文件内容
                try:
                    # 先尝试UTF-8编码
                    content_str = file_content.decode('utf-8')
                except UnicodeDecodeError:
                    # 如果UTF-8失败，尝试其他常见编码
                    try:
                        content_str = file_content.decode('latin1')
                    except Exception as decode_error:
                        return JsonResponse({
                            'status': 'error',
                            'title': '文件解码失败',
                            'message': f'无法解码文件内容。请确保使用UTF-8或Latin1编码。错误: {str(decode_error)}'
                        })
                
                # 使用StringIO创建文件类对象
                try:
                    csv_file = io.StringIO(content_str)
                    df = pd.read_csv(
                        csv_file,
                        sep=None,
                        engine='python',
                        header=0,
                        skipinitialspace=True,
                        skip_blank_lines=True,
                        on_bad_lines='warn'
                    )
                except Exception as read_error:
                    return JsonResponse({
                        'status': 'error',
                        'title': 'CSV解析失败',
                        'message': f'无法解析CSV文件。请检查文件格式。错误: {str(read_error)}'
                    })

                if df.empty:
                    return JsonResponse({
                        'status': 'error',
                        'title': '文件内容为空',
                        'message': '上传的文件不包含任何数据'
                    })

                # 验证字段
                model_fields = QAR.get_fields()
                missing_fields = set(model_fields) - set(df.columns)
                if missing_fields:
                    return JsonResponse({
                        'status': 'error',
                        'title': '字段不匹配',
                        'message': f'缺少必要字段: {", ".join(missing_fields)}'
                    })

                # 准备批量插入数据
                objects_to_create = []
                error_lines = []
                
                for index, row in df.iterrows():
                    try:
                        data_dict = {'qar_id': qar_id, 'label': flight_label}
                        
                        for field in model_fields:
                            if field in ['qar_id', 'label']:
                                continue
                                
                            value = row[field]
                            if pd.isna(value):
                                data_dict[field] = None
                            else:
                                model_field = QAR._meta.get_field(field)
                                if model_field.get_internal_type() in ['FloatField', 'IntegerField']:
                                    try:
                                        data_dict[field] = float(value)
                                    except ValueError:
                                        error_lines.append(f"行 {index+2}: 字段 '{field}' 的值 '{value}' 无法转换为数字")
                                        continue
                                else:
                                    data_dict[field] = str(value)
                        
                        objects_to_create.append(QAR(**data_dict))
                        
                    except Exception as row_error:
                        error_lines.append(f"行 {index+2}: {str(row_error)}")
                
                if error_lines:
                    return JsonResponse({
                        'status': 'error',
                        'title': '数据格式错误',
                        'message': '发现以下数据问题:\n' + '\n'.join(error_lines[:10]) + 
                                 ('\n...(仅显示前10条错误)' if len(error_lines) > 10 else '')
                    })

                # 批量插入数据
                try:
                    with transaction.atomic():
                        qar_instances = QAR.objects.bulk_create(objects_to_create, batch_size=1000)
                        # 手动触发统计更新（全量更新）
                        update_qar_parameter_stats(sender=QAR, instance=None)
                        # 手动更新QAR_Mask表
                        update_qar_mask(sender=QAR, instance=qar_instances)
                        # 更新Flight_Trajectory表
                        update_flight_trajectory(qar_instances)

                        success_count = len(objects_to_create)
                        
                        label_text = dict(DataUploadForm.LABEL_CHOICES).get(int(flight_label), '未知')
                        return JsonResponse({
                            'status': 'success',
                            'title': '上传成功',
                            'message': f'成功导入 {success_count} 条记录\nQAR ID: {qar_id}\n标签: {label_text}'
                        })
                        
                except Exception as db_error:
                    return JsonResponse({
                        'status': 'error',
                        'title': '数据库错误',
                        'message': f'数据保存失败: {str(db_error)}'
                    })
            
            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'title': '处理错误',
                    'message': f'处理文件时出错: {str(e)}'
                })
        else:
            errors = '\n'.join([f"{field}: {error[0]}" for field, error in form.errors.items()])
            return JsonResponse({
                'status': 'error',
                'title': '表单验证失败',
                'message': errors
            })
    
    # GET请求处理
    form = DataUploadForm()
    return render(request, 'data_management/data_upload.html', {
        'form': form,
        'model_fields': QAR.get_fields()
    })

def data_preview(request):
    # 获取所有QAR ID并按首字母排序
    qar_ids = QAR.objects.order_by('qar_id').values_list('qar_id', flat=True).distinct()

    # 获取字段列表作为表头（自动匹配模型字段）
    model_fields = [field.verbose_name for field in QAR._meta.fields]
    field_names = [field.name for field in QAR._meta.fields]

    # 获取页码参数，默认第一页
    page_number = request.GET.get('page', 1)
    per_page = 50  # 每页显示数量

    # 获取查询参数
    qar_id = request.GET.get('qar_id', '')

    # 构建查询
    query_set = QAR.objects.all()
    
    # 如果提供了QARID，添加过滤条件
    if qar_id:
        query_set = query_set.filter(qar_id=qar_id)

    # 查询字段值（values生成dict更方便）
    query_set = query_set.values(*field_names)
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
        'page_obj': page_obj,
        'qar_id': qar_id,  # 传递查询参数到模板
        'qar_ids': qar_ids,  # 所有QAR ID列表
    }

    return render(request, 'data_management/data_preview.html', context)


def data_monitor(request):
    # 1. 计算不同飞行状态的数据缺失率
    flight_status_stats = calculate_flight_status_stats()
    
    # 2. 计算各字段的缺失率统计
    field_stats = calculate_field_missing_stats()

    context = {
        'flight_status_stats': flight_status_stats,
        'field_stats': field_stats,
    }
    
    return render(request, 'data_management/data_monitor.html', context)



def data_imputation(request):
    # 初始化配置
    config = LGTDMConfig()
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')
    
    # 属性无关字段
    EXCLUDE_FIELDS = {'id', 'qar_id', 'label'}
    
    # 获取所有QAR ID用于选择
    qar_ids = QAR.objects.values_list('qar_id', flat=True).distinct()
    selected_qar_id = request.GET.get('qar_id') or request.POST.get('qar_id')
    if not selected_qar_id:
        qar_ids = list(qar_ids)
        selected_qar_id = qar_ids[0]
    
    # 获取LGTDM模型输入字段
    input_fields = QAR_Parameter_Attribute.objects.filter(
        normalized_variance__gt=0.1
    ).exclude(
        parameter_name__in=EXCLUDE_FIELDS
    ).values_list('parameter_name', flat=True)
    
    # 获取统计信息
    stats = QAR_Parameter_Attribute.objects.filter(
        parameter_name__in=input_fields
    ).values('parameter_name', 'mean_value', 'variance')
    
    stats_dict = {
        item['parameter_name']: {
            'mean': item['mean_value'],
            'std': item['variance'] ** 0.5
        }
        for item in stats
    }
    
    # 获取字段信息
    model_fields = [field.verbose_name for field in QAR._meta.fields]
    field_names = [field.name for field in QAR._meta.fields]
    
    # 分页处理
    page_number = request.GET.get('page', 1)
    per_page = 150
    
    # 根据选择的QAR ID筛选数据
    query_set = QAR.objects.filter(qar_id=selected_qar_id).values(*field_names) if selected_qar_id else QAR.objects.values(*field_names)
    paginator = Paginator(query_set, per_page)
    
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    
    # 转换为DataFrame
    df = pd.DataFrame(list(page_obj.object_list), columns=field_names)
    
    if request.method == 'GET':
        # 获取完整的QAR_Mask数据（不分页）
        full_mask_data = QAR_Mask.objects.filter(qar_id=selected_qar_id).values(*[f"{field}_mask" for field in field_names if field not in EXCLUDE_FIELDS])
        
        df_t = df.copy()

        if full_mask_data and not df_t.empty:
            # 将掩码数据转换为DataFrame
            full_mask_df = pd.DataFrame(list(full_mask_data))
            
            # 计算当前分页对应的掩码数据范围
            start_idx = (page_obj.number - 1) * per_page
            end_idx = start_idx + len(df_t)
            
            # 获取当前页对应的掩码数据
            mask_df = full_mask_df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # 确保当前页的QAR和QAR_Mask行数匹配
            if len(df_t) == len(mask_df):
                # 应用掩码到每个字段
                for field in field_names:
                    if field not in EXCLUDE_FIELDS:
                        mask_field = f"{field}_mask"
                        if mask_field in mask_df.columns:
                            # 只对掩码为False(0)的位置应用NaN
                            df_t.loc[~mask_df[mask_field].astype(bool), field] = 'NaN'
        
        # 准备返回数据
        table_data = df_t.values.tolist()
        
        context = {
            'table_data': table_data,
            'table_headings': model_fields,
            'page_obj': page_obj,
            'is_masked': True,
            'qar_ids': qar_ids,
            'selected_qar_id': selected_qar_id,
        }
        
    elif request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'repair_data':
            # 加载模型
            seq_dim, num_label = get_data_info()
            model = LGTDM.Model(config, seq_dim, num_label, device)
            model.load_state_dict(torch.load('apps/ml_models/LGTDM/save/model_weights.pth'))
            model.to(device)
            model.eval()
            
            original_df = df.copy()

            # 获取完整的QAR_Mask数据（不分页）
            full_mask_data = QAR_Mask.objects.filter(qar_id=selected_qar_id).values(*[f"{field}_mask" for field in input_fields])
            # 将掩码数据转换为DataFrame
            full_mask_df = pd.DataFrame(list(full_mask_data))
            # 计算当前分页对应的掩码数据范围
            start_idx = (page_obj.number - 1) * per_page
            end_idx = start_idx + len(df)
            
            # 获取当前页对应的掩码数据
            mask_df = full_mask_df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # 数据标准化
            for field in input_fields:
                if field in df.columns and field in stats_dict:
                    mean = stats_dict[field]['mean']
                    std = stats_dict[field]['std']
                    df[field] = (df[field] - mean) / std
            
            # 准备模型输入
            data = df[input_fields].fillna(0).values.astype(np.float32)
            obs_mask = mask_df.to_numpy().astype(np.float32)
            
            # 转换为张量
            data = torch.from_numpy(data).unsqueeze(0).to(device)
            obs_mask = torch.from_numpy(obs_mask).unsqueeze(0).to(device)
            label = torch.tensor([df['label'].iloc[0]]).to(device)
            
            # 调用模型填充缺失值
            with torch.no_grad():
                imputation, _, _ = model.forward(
                    mode='test',
                    input_data=(data, obs_mask, obs_mask),  # 使用相同mask
                    label=label
                )
            
            # 处理填充结果
            imputation_data = imputation.squeeze().cpu().numpy()
            result_df = original_df.copy()
            
            for i, field in enumerate(input_fields):
                if field in stats_dict:
                    mean = stats_dict[field]['mean']
                    std = stats_dict[field]['std']
                    # 修改为检测0值而不是NaN
                    mask_positions = (df[field] == 0)  # 检测值为0的位置
                    # 应用修复：只修复值为0的位置
                    result_df.loc[mask_positions, field] = imputation_data[mask_positions, i] * std + mean
            
            table_data = result_df.values.tolist()
            
            context = {
                'table_data': table_data,
                'table_headings': model_fields,
                'page_obj': page_obj,
                'is_filled': True,
                'qar_ids': qar_ids,
                'selected_qar_id': selected_qar_id,
            }
            
        elif action == 'train_model':
            # 调用训练函数
            result = run_lgtdm(is_training=True)
            return redirect('data_imputation')
    
    return render(request, 'data_management/data_imputation.html', context)


def data_imputation1(request):
    # 初始化配置
    config = LGTDMConfig()
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')
    
    # 属性无关字段
    EXCLUDE_FIELDS = {'id', 'qar_id', 'label'}
    
    # 获取所有QAR ID用于选择
    qar_ids = QAR.objects.values_list('qar_id', flat=True).distinct()
    selected_qar_id = request.GET.get('qar_id') or request.POST.get('qar_id')
    if not selected_qar_id:
        qar_ids = list(qar_ids)
        selected_qar_id = qar_ids[0]
    
    # 获取LGTDM模型输入字段
    input_fields = QAR_Parameter_Attribute.objects.filter(
        normalized_variance__gt=0.1
    ).exclude(
        parameter_name__in=EXCLUDE_FIELDS
    ).values_list('parameter_name', flat=True)
    
    # 获取统计信息
    stats = QAR_Parameter_Attribute.objects.filter(
        parameter_name__in=input_fields
    ).values('parameter_name', 'mean_value', 'variance')
    
    stats_dict = {
        item['parameter_name']: {
            'mean': item['mean_value'],
            'std': item['variance'] ** 0.5
        }
        for item in stats
    }
    
    # 获取字段信息
    model_fields = [field.verbose_name for field in QAR._meta.fields]
    field_names = [field.name for field in QAR._meta.fields]
    
    # 分页处理
    page_number = request.GET.get('page', 1)
    per_page = 150
    
    # 根据选择的QAR ID筛选数据
    query_set = QAR.objects.filter(qar_id=selected_qar_id).values(*field_names) if selected_qar_id else QAR.objects.values(*field_names)
    paginator = Paginator(query_set, per_page)
    
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    
    # 转换为DataFrame
    df = pd.DataFrame(list(page_obj.object_list), columns=field_names)
    
    if request.method == 'GET':
        # 获取完整的QAR_Mask数据（不分页）
        full_mask_data = QAR_Mask.objects.filter(qar_id=selected_qar_id).values(*[f"{field}_mask" for field in field_names if field not in EXCLUDE_FIELDS])
        
        df_t = df.copy()
        mask_positions = {}  # 存储每个字段的缺失位置

        if full_mask_data and not df_t.empty:
            # 将掩码数据转换为DataFrame
            full_mask_df = pd.DataFrame(list(full_mask_data))
            
            # 计算当前分页对应的掩码数据范围
            start_idx = (page_obj.number - 1) * per_page
            end_idx = start_idx + len(df_t)
            
            # 获取当前页对应的掩码数据
            mask_df = full_mask_df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # 确保当前页的QAR和QAR_Mask行数匹配
            if len(df_t) == len(mask_df):
                # 应用掩码到每个字段
                for field in field_names:
                    if field not in EXCLUDE_FIELDS:
                        mask_field = f"{field}_mask"
                        if mask_field in mask_df.columns:
                            # 记录缺失位置
                            mask_positions[field] = ~mask_df[mask_field].astype(bool)
                            # 只对掩码为False(0)的位置应用NaN
                            df_t.loc[mask_positions[field], field] = 'NaN'
        
        # 准备返回数据
        table_data = df_t.values.tolist()
        
        context = {
            'table_data': table_data,
            'table_headings': model_fields,
            'page_obj': page_obj,
            'is_masked': True,
            'qar_ids': qar_ids,
            'selected_qar_id': selected_qar_id,
            'mask_positions': mask_positions,  # 添加缺失位置信息
        }
        
    elif request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'repair_data':
            # 加载模型
            seq_dim, num_label = get_data_info()
            model = LGTDM.Model(config, seq_dim, num_label, device)
            model.load_state_dict(torch.load('apps/ml_models/LGTDM/save/model_weights.pth'))
            model.to(device)
            model.eval()
            
            original_df = df.copy()
            mask_positions = {}  # 存储每个字段的缺失位置

            # 获取完整的QAR_Mask数据（不分页）
            full_mask_data = QAR_Mask.objects.filter(qar_id=selected_qar_id).values(*[f"{field}_mask" for field in input_fields])
            # 将掩码数据转换为DataFrame
            full_mask_df = pd.DataFrame(list(full_mask_data))
            # 计算当前分页对应的掩码数据范围
            start_idx = (page_obj.number - 1) * per_page
            end_idx = start_idx + len(df)
            
            # 获取当前页对应的掩码数据
            mask_df = full_mask_df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # 数据标准化
            for field in input_fields:
                if field in df.columns and field in stats_dict:
                    mean = stats_dict[field]['mean']
                    std = stats_dict[field]['std']
                    df[field] = (df[field] - mean) / std
            
            # 准备模型输入
            data = df[input_fields].fillna(0).values.astype(np.float32)
            obs_mask = mask_df.to_numpy().astype(np.float32)
            
            # 转换为张量
            data = torch.from_numpy(data).unsqueeze(0).to(device)
            obs_mask = torch.from_numpy(obs_mask).unsqueeze(0).to(device)
            label = torch.tensor([df['label'].iloc[0]]).to(device)
            
            # 调用模型填充缺失值
            with torch.no_grad():
                imputation, _, _ = model.forward(
                    mode='test',
                    input_data=(data, obs_mask, obs_mask),  # 使用相同mask
                    label=label
                )
            
            # 处理填充结果
            imputation_data = imputation.squeeze().cpu().numpy()
            result_df = original_df.copy()
            
            for i, field in enumerate(input_fields):
                if field in stats_dict:
                    mean = stats_dict[field]['mean']
                    std = stats_dict[field]['std']
                    # 记录缺失位置
                    mask_field = f"{field}_mask"
                    if mask_field in mask_df.columns:
                        mask_positions[field] = ~mask_df[mask_field].astype(bool)
                    # 应用修复：只修复值为0的位置
                    result_df.loc[mask_positions[field], field] = imputation_data[mask_positions[field], i] * std + mean
            
            table_data = result_df.values.tolist()
            
            context = {
                'table_data': table_data,
                'table_headings': model_fields,
                'page_obj': page_obj,
                'is_filled': True,
                'qar_ids': qar_ids,
                'selected_qar_id': selected_qar_id,
                'mask_positions': mask_positions,  # 添加缺失位置信息
            }
            
        elif action == 'train_model':
            # 调用训练函数
            result = run_lgtdm(is_training=True)
            return redirect('data_imputation')
    
    return render(request, 'data_management/data_imputation1.html', context)