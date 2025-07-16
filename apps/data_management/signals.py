from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.db import transaction
from .models import QAR_Mask, QAR_Label_Missing_Stats
from apps.core.models import QAR
from django.utils import timezone
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


# 创建或更新QAR_Mask记录
def update_qar_mask(sender, instance=None, **kwargs):
    qar_data = {qar.id: qar.qar_id for qar in instance}

    # 构建QAR_Mask记录
    mask_objs = []
    for qar in instance:
        mask_data = {"id": qar.id, "qar_id": qar.qar_id}
        
        # 遍历所有_mask字段，检查QAR字段是否为NULL
        for field in QAR_Mask._meta.get_fields():
            if not field.name.endswith('_mask') or field.name == 'id':
                continue
            
            qar_field = field.name.replace('_mask', '')
            mask_data[field.name] = 1 if getattr(qar, qar_field) is not None else 0
        
        mask_objs.append(QAR_Mask(**mask_data))

    # 批量插入QAR_Mask记录
    QAR_Mask.objects.bulk_create(mask_objs, batch_size=1000)


# 创新或更新QAR_Label_Missing_Stats记录
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
                    'missing_fields_count': stats['missing'],
                    'completeness_rate': round(completeness, 2)
                }
            )
    
    return True