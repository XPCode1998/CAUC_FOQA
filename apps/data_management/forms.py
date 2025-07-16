from django import forms
from django.core.validators import FileExtensionValidator
import re

class DataUploadForm(forms.Form):
    LABEL_CHOICES = [
        (0, '正常飞行'),
        (1, '结冰'),
        (2, '单发失效'),
        (3, '双发失效'),
        (4, '低能量'),
    ]
    
    label = forms.ChoiceField(
        label='飞行风险标签',
        choices=LABEL_CHOICES,
        initial=0,
        widget=forms.Select(attrs={
            'class': 'form-select',
            'required': 'required'
        }))
    
    file = forms.FileField(
        label='QAR数据文件',
        validators=[FileExtensionValidator(allowed_extensions=['txt', 'csv'])],
        widget=forms.FileInput(attrs={
            'accept': '.txt,.csv',
            'class': 'hidden',
            'id': 'qarFileInput',
            'required': 'required'
        }))
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if not file:
            return file
        
        # 1. 验证文件大小
        if file.size > 10 * 1024 * 1024:  # 10MB限制
            raise forms.ValidationError("文件大小不能超过10MB")
        
        # 2. 验证文件名格式（YYYY-MM-DD_ID.csv）
        filename = file.name
        match = re.match(r'^(\d{4})-(\d{2})-(\d{2})_(\d+)\.(csv|txt)$', filename)
        if not match:
            raise forms.ValidationError(
                "文件名格式应为 YYYY-MM-DD_ID.csv（例如：2023-08-31_24440.csv）"
            )
        
        # 3. 提取 QAR ID（2023083124440）
        year, month, day, qar_id, _ = match.groups()
        self.cleaned_data['qar_id'] = f"{year}{month}{day}{qar_id}"
        
        return file