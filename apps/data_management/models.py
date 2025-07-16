from django.db import models

# Create your models here.
class QAR_Mask(models.Model):
    """与QAR表同结构(缺少标签列）的缺失掩码表, 每个字段对应QAR表中相同字段的缺失状态"""
    qar_id = models.CharField('QAR ID', max_length=60)
    dSimTime_mask = models.BooleanField('模拟时间掩码', default=False)
    dStepTime_mask = models.BooleanField('单步运行模拟时间掩码', default=False)
    dGravityAcc_mask = models.BooleanField('重力加速度掩码', default=False)
    dUwg_mask = models.BooleanField('风速沿地轴X轴分量掩码', default=False)
    dVwg_mask = models.BooleanField('风速沿地轴Y轴分量掩码', default=False)
    dWwg_mask = models.BooleanField('风速沿地轴Z轴分量掩码', default=False)
    dUTrub_mask = models.BooleanField('紊流沿机体轴X轴分量掩码', default=False)
    dVTrub_mask = models.BooleanField('紊流沿机体轴Y轴分量掩码', default=False)
    dWTrub_mask = models.BooleanField('紊流沿机体轴Z轴分量掩码', default=False)
    dMass_mask = models.BooleanField('飞机质量掩码', default=False)
    dAlpha_mask = models.BooleanField('迎角掩码', default=False)
    dAlphaRad_mask = models.BooleanField('迎角(弧度)掩码', default=False)
    dSinAlpha_mask = models.BooleanField('迎角正弦值掩码', default=False)
    dCosAlpha_mask = models.BooleanField('迎角余弦值掩码', default=False)
    dAlphaDot_mask = models.BooleanField('迎角变化率掩码', default=False)
    dBeta_mask = models.BooleanField('侧滑角掩码', default=False)
    dBetaRad_mask = models.BooleanField('侧滑角(弧度)掩码', default=False)
    dSinBeta_mask = models.BooleanField('侧滑角正弦值掩码', default=False)
    dCosBeta_mask = models.BooleanField('侧滑角余弦值掩码', default=False)
    dBetaDot_mask = models.BooleanField('侧滑角变化率掩码', default=False)
    dPhi_mask = models.BooleanField('滚转角掩码', default=False)
    dTheta_mask = models.BooleanField('俯仰角掩码', default=False)
    dPsi_mask = models.BooleanField('偏航角掩码', default=False)
    dChi_mask = models.BooleanField('航迹方位角掩码', default=False)
    dGamma_mask = models.BooleanField('航迹爬升角掩码', default=False)
    dGroundspeed_mask = models.BooleanField('航迹速度掩码', default=False)
    dTAS_mask = models.BooleanField('真空速掩码', default=False)
    dMach_mask = models.BooleanField('马赫数掩码', default=False)
    dPd_mask = models.BooleanField('动压掩码', default=False)
    dPs_mask = models.BooleanField('静压掩码', default=False)
    dUk_mask = models.BooleanField('航迹速度X分量掩码', default=False)
    dVk_mask = models.BooleanField('航迹速度Y分量掩码', default=False)
    dWk_mask = models.BooleanField('航迹速度Z分量掩码', default=False)
    dUkDot_mask = models.BooleanField('线加速度X分量掩码', default=False)
    dVkDot_mask = models.BooleanField('线加速度Y分量掩码', default=False)
    dWkDot_mask = models.BooleanField('线加速度Z分量掩码', default=False)
    dU_mask = models.BooleanField('空速X分量掩码', default=False)
    dV_mask = models.BooleanField('空速Y分量掩码', default=False)
    dW_mask = models.BooleanField('空速Z分量掩码', default=False)
    dP_mask = models.BooleanField('机体轴X轴角速度掩码', default=False)
    dQ_mask = models.BooleanField('机体轴Y轴角速度掩码', default=False)
    dR_mask = models.BooleanField('机体轴Z轴角速度掩码', default=False)
    dPRad_mask = models.BooleanField('机体轴X轴角速度(弧度)掩码', default=False)
    dQRad_mask = models.BooleanField('机体轴Y轴角速度(弧度)掩码', default=False)
    dRRad_mask = models.BooleanField('机体轴Z轴角速度(弧度)掩码', default=False)
    dRDot_mask = models.BooleanField('机体轴X轴角加速度掩码', default=False)
    dPDot_mask = models.BooleanField('机体轴Y轴角加速度掩码', default=False)
    dQDot_mask = models.BooleanField('机体轴Z轴角加速度掩码', default=False)
    dNx_mask = models.BooleanField('X轴过载掩码', default=False)
    dNy_mask = models.BooleanField('Y轴过载掩码', default=False)
    dNz_mask = models.BooleanField('Z轴过载掩码', default=False)
    dUkgDot_mask = models.BooleanField('线加速度地轴X分量掩码', default=False)
    dVkgDot_mask = models.BooleanField('线加速度地轴Y分量掩码', default=False)
    dWkgDot_mask = models.BooleanField('线加速度地轴Z分量掩码', default=False)
    dUkg_mask = models.BooleanField('航迹速度地轴X分量掩码', default=False)
    dVkg_mask = models.BooleanField('航迹速度地轴Y分量掩码', default=False)
    dWkg_mask = models.BooleanField('航迹速度地轴Z分量掩码', default=False)
    dLongitude_mask = models.BooleanField('经度掩码', default=False)
    dLatitude_mask = models.BooleanField('纬度掩码', default=False)
    dASL_mask = models.BooleanField('海拔高度掩码', default=False)
    dAGL_mask = models.BooleanField('离地高度掩码', default=False)
    dTrueHeading_mask = models.BooleanField('真航向掩码', default=False)
    dMagHeading_mask = models.BooleanField('磁航向掩码', default=False)
    dPosXg_mask = models.BooleanField('地轴X坐标掩码', default=False)
    dPosYg_mask = models.BooleanField('地轴Y坐标掩码', default=False)
    dPosZg_mask = models.BooleanField('地轴Z坐标掩码', default=False)
    dtx_mask = models.BooleanField('副翼偏度掩码', default=False)
    dty_mask = models.BooleanField('方向舵偏度掩码', default=False)
    dtz_mask = models.BooleanField('升降舵偏度掩码', default=False)
    LGPos_mask = models.BooleanField('起落架位置掩码', default=False)
    dFlap_mask = models.BooleanField('襟翼偏度掩码', default=False)
    gfuel_mask = models.BooleanField('剩余油量掩码', default=False)
    pe_t1_mask = models.BooleanField('油门杆位置1掩码', default=False)
    pe_t2_mask = models.BooleanField('油门杆位置2掩码', default=False)
    rot1_mask = models.BooleanField('发动机转速1掩码', default=False)
    rot2_mask = models.BooleanField('发动机转速2掩码', default=False)
    thrust_mask = models.BooleanField('发动机推力掩码', default=False)
    gfused_mask = models.BooleanField('耗油量掩码', default=False)
    dGtNormal_mask = models.BooleanField('燃油消耗率掩码', default=False)
    dGfNormal_mask = models.BooleanField('燃油消耗量掩码', default=False)
    dGFuel_mask = models.BooleanField('余油量掩码', default=False)
    dDa_mask = models.BooleanField('滚转操纵位置掩码', default=False)
    dDr_mask = models.BooleanField('偏航操纵位置掩码', default=False)
    dDe_mask = models.BooleanField('俯仰操纵位置掩码', default=False)
    dDaTrim_mask = models.BooleanField('配平滚转操纵量掩码', default=False)
    dDeTrim_mask = models.BooleanField('配平俯仰操纵量掩码', default=False)

    class Meta:
        verbose_name = 'QAR缺失掩码表'
        verbose_name_plural = 'QAR缺失掩码表'

    def __str__(self):
        return f"{self.qar_id} 缺失掩码"
    
class QAR_Label_Missing_Stats(models.Model):
    label = models.IntegerField('飞行状态标签', choices=[
        (0, '正常状态'),
        (1, '结冰状态'),
        (2, '单发失效'), 
        (3, '双发失效'),
        (4, '低能量')
    ])
    total_records = models.IntegerField('总记录数', default=0)
    missing_fields_count = models.IntegerField('缺失字段总数', default=0)
    completeness_rate = models.FloatField('完整率(%)', default=100.0)
    last_updated = models.DateTimeField('最后更新时间', auto_now=True)

    class Meta:
        verbose_name = '标签缺失统计'
        verbose_name_plural = '标签缺失统计'
        unique_together = ('label',)
        
    def __str__(self):
        return f"{self.get_label_display()} (完整率: {self.completeness_rate:.2f}%)"