# utils/stats_updater.py
from django.db import transaction
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from .models import QAR_Mask, QAR_Label_Missing_Stats
from apps.core.models import QAR, QAR_Parameter_Attribute
from django.db import transaction
from apps.core.models import QAR, QAR_Parameter_Attribute
from .models import QAR_Mask
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


def update_label_missing_stats():
    """更新所有标签的缺失统计数据"""
    # 1. 获取所有掩码字段名
    mask_fields = [f.name for f in QAR_Mask._meta.get_fields() 
                 if f.name.endswith('_mask') and f.name not in ['id', 'qar_id']]
    
    # 2. 批量获取QAR标签映射
    qar_labels = dict(QAR.objects.values_list('qar_id', 'label'))
    
    # 3. 批量获取掩码数据
    mask_values = QAR_Mask.objects.values('qar_id', *mask_fields)
    
    # 4. 并行计算各标签的缺失情况
    label_stats = defaultdict(lambda: {'total': 0, 'missing': 0})
    
    def process_mask(mask):
        label = qar_labels.get(mask['qar_id'], 0)
        missing = sum(1 for field in mask_fields if mask[field])
        return (label, missing)
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_mask, mask_values)
        for label, missing in results:
            label_stats[label]['total'] += 1
            label_stats[label]['missing'] += missing
    
    # 5. 计算并保存统计结果
    with transaction.atomic():
        for label, stats in label_stats.items():
            total_records = stats['total']
            total_fields = total_records * len(mask_fields)
            completeness = 100 - (stats['missing'] / total_fields * 100) if total_fields > 0 else 100
            
            QAR_Label_Missing_Stats.objects.update_or_create(
                label=label,
                defaults={
                    'total_records': total_records,
                    'missing_fields_count': total_fields - stats['missing'],
                    'completeness_rate': 100 - round(completeness, 2)
                }
            )
    
    return True



def calculate_flight_status_stats():
    """
    从 QAR_Label_Missing_Stats 表获取各飞行状态的缺失统计数据
    返回格式与原来保持一致，便于前端兼容
    """
    # 直接从预计算的统计表获取数据
    stats = QAR_Label_Missing_Stats.objects.all()
    
    status_stats = []
    for stat in stats:
        missing_rate = 100 - stat.completeness_rate
        status_class = 'danger' if missing_rate > 20 else \
                     'warning' if missing_rate > 10 else 'success'
        
        status_stats.append({
            'name': stat.get_label_display(),
            'missing_rate': round(missing_rate, 2),
            'completeness': round(stat.completeness_rate, 2),
            'total': stat.total_records,
            'status_class': status_class,
        })
    
    # 确保所有标签都有数据（处理新增标签情况）
    existing_labels = {s.label for s in stats}
    STATUS_LABELS = {
        0: '正常状态',
        1: '结冰状态',
        2: '单发失效',
        3: '双发失效',
        4: '低能量',
    }
    
    for label, name in STATUS_LABELS.items():
        if label not in existing_labels:
            status_stats.append({
                'name': name,
                'missing_rate': 0.0,
                'completeness': 100.0,
                'total': 0,
                'status_class': 'success',
            })
    
    # 按标签排序保持一致性
    status_stats.sort(key=lambda x: list(STATUS_LABELS.values()).index(x['name']))
    
    return status_stats


def calculate_field_missing_stats():
    """
    计算字段缺失统计，仅包含 QAR_Parameter_Attribute 中 normalized_variance > 0.1 的字段
    返回格式保持与原来一致
    """
    # 1. 预计算总记录数 (单次查询)
    total = QAR.objects.count() or 1
    
    # 2. 获取需要统计的字段（normalized_variance > 0.1 且 is_monitored=True）
    parameter_fields = QAR_Parameter_Attribute.objects.filter(
        normalized_variance__gt=0.1,
        is_monitored=True
    ).values_list('parameter_name', flat=True)
    
    # 3. 获取这些字段对应的模型字段对象
    qar_fields = [
        field for field in QAR._meta.get_fields() 
        if field.name in parameter_fields and 
           field.name not in ['id', 'label', 'qar_id']
    ]
    
    # 4. 批量获取所有掩码统计
    field_stats = []
    
    def process_field(field):
        field_name = field.name
        mask_field = f"{field_name}_mask"
        
        if not hasattr(QAR_Mask, mask_field):
            return None
        
        # 获取参数属性信息（单次查询）
        param_attr = QAR_Parameter_Attribute.objects.get(parameter_name=field_name)
        
        # 计算缺失记录数 (单次查询)
        missing = total - QAR_Mask.objects.filter(**{mask_field: True}).count() or 0
        
        # 计算缺失率和完整性
        missing_rate = round((missing / total) * 100, 2) if total > 0 else 0
        completeness = 100 - missing_rate
        
        # 确定状态颜色
        if missing_rate > 20:
            status_class = 'danger'
        elif missing_rate > 10:
            status_class = 'warning'
        elif missing_rate > 5:
            status_class = 'primary'
        else:
            status_class = 'success'
        
        return {
            'name': field_name,
            'verbose_name': param_attr.description or field_name,
            'missing_rate': missing_rate,
            'completeness': completeness,
            'missing_count': missing,
            'status_class': status_class,
            'variance': param_attr.variance,
            'unit': param_attr.unit,
        }
    
    # 使用线程池并行处理字段
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_field, qar_fields)
        field_stats = [result for result in results if result is not None]
    
    # 按缺失率降序排序
    field_stats.sort(key=lambda x: x['missing_rate'], reverse=True)
    
    return field_stats