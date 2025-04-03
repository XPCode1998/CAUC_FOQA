from rest_framework import serializers
from .models import QAR

class QARSerializer(serializers.ModelSerializer):
    class Meta:
        model = QAR
        fields = '__all__'  # 序列化所有字段
        