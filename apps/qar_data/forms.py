from django import forms
from django.core.validators import FileExtensionValidator

class QARUploadForm(forms.Form):
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
        help_text="请选择本次飞行的风险类别",
        widget=forms.Select(attrs={
            'class': 'form-select',
            'x-model': 'flightLabel',
            'required': 'required'
        }))
    
    file = forms.FileField(
        label='QAR数据文件',
        help_text="请上传.txt或.csv格式的QAR数据文件",
        validators=[FileExtensionValidator(allowed_extensions=['txt', 'csv'])],
        widget=forms.FileInput(attrs={
            'accept': '.txt,.csv',
            'class': 'hidden',
            'id': 'qarFileInput',
            'required': 'required',
            'x-on:change': 'previewFile'
        }))
    
    qar_id = forms.CharField(
        required=False,
        widget=forms.HiddenInput(attrs={
            'x-bind:value': 'qarId'
        }))
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            if file.size > 10 * 1024 * 1024:  # 10MB限制
                raise forms.ValidationError("文件大小不能超过10MB")
            if not (file.name.lower().endswith('.txt') or file.name.lower().endswith('.csv')):
                raise forms.ValidationError("支持.txt和.csv格式的文件")
        return file