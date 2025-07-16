from django.db import models

# Create your models here.
class Flight_Trajectory(models.Model):
    qar_id = models.CharField('QAR ID', max_length=60, null=True)
    dSimTime = models.FloatField('模拟时间', null=True)
    dTrueHeading = models.FloatField('真航向', null=True)
    dMagHeading = models.FloatField('磁航向', null=True)
    dLongitude = models.FloatField('经度', null=True)
    dLatitude = models.FloatField('纬度', null=True)
    dASL = models.FloatField('飞机重心海拔高度', null=True)
    dAGL = models.FloatField('飞机重心离地高度', null=True)
    dLongitude_change = models.FloatField('经度变化量', null=True)
    dLatitude_change = models.FloatField('纬度变化量', null=True)
    dASL_change = models.FloatField('飞机重心海拔高度变化量', null=True)
    dAGL_change = models.FloatField('飞机重心离地高度变化量', null=True)
    
    
    class Meta:
        verbose_name = '航迹'
        verbose_name_plural = '航迹'