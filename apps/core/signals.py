from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.db import models, transaction
from django.db.models import Min, Max, Avg, Variance
from .models import QAR, QAR_Parameter_Attribute
import numpy as np
import logging

logger = logging.getLogger(__name__)

# @receiver(post_save, sender=QAR)
# @receiver(post_delete, sender=QAR)
def update_qar_parameter_stats(sender, instance=None, **kwargs):
    """
    在QAR模型保存或删除后更新QAR_Parameter_Attribute中的统计信息
    自动设置警告阈值(90%分位数)和严重阈值(95%分位数)
    """
    try:
        with transaction.atomic():
            # 获取所有数值型字段
            numeric_fields = [
                field for field in QAR._meta.get_fields()
                if isinstance(field, (models.FloatField, models.IntegerField))
                and field.name in QAR.get_fields()
            ]
            
            if not numeric_fields:
                logger.info("QAR模型中没有找到数值型字段")
                return
                
            for field in numeric_fields:
                field_name = field.name
                
                # 获取非空值并按升序排序
                values = list(QAR.objects.filter(**{f"{field_name}__isnull": False})
                             .order_by(field_name)
                             .values_list(field_name, flat=True))
                
                if not values:
                    logger.debug(f"字段 {field_name} 没有有效数据，跳过统计计算")
                    continue
                
                # 计算基本统计量
                stats = {
                    'min_val': min(values),
                    'max_val': max(values),
                    'mean_val': np.mean(values),
                    'var_val': np.var(values),
                    'median': np.median(values),
                    'q1': np.percentile(values, 25),
                    'q3': np.percentile(values, 75),
                    'warning_lower': np.percentile(values, 5),   # 5%作为警告下限
                    'warning_upper': np.percentile(values, 90),  # 90%作为警告上限
                    'critical_lower': np.percentile(values, 2.5), # 2.5%作为严重下限
                    'critical_upper': np.percentile(values, 95)  # 95%作为严重上限
                }
                
                # 计算归一化方差
                normalized_var = stats['var_val'] / stats['mean_val'] if stats['mean_val'] != 0 else 0
                # 更新或创建统计记录
                QAR_Parameter_Attribute.objects.update_or_create(
                    parameter_name=field_name,
                    defaults={
                        'description': field.verbose_name,
                        'unit': getattr(field, 'unit', ''),
                        'min_value': stats['min_val'],
                        'max_value': stats['max_val'],
                        'mean_value': stats['mean_val'],
                        'variance': stats['var_val'],
                        'normalized_variance': normalized_var,
                        'warning_lower': stats['warning_lower'],
                        'warning_upper': stats['warning_upper'],
                        'critical_lower': stats['critical_lower'],
                        'critical_upper': stats['critical_upper'],
                        'is_monitored': True  # 默认启用监控
                    }
                )
                
                logger.info(f"成功更新字段 {field_name} 的统计信息和阈值设置")
                
    except Exception as e:
        logger.error(f"更新QAR参数统计信息时出错: {str(e)}", exc_info=True)
        # 在实际应用中，可能需要添加错误通知机制