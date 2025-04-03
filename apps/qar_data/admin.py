from django.contrib import admin
from .models import QAR
from import_export import resources
from import_export.admin import ImportExportModelAdmin
# Register your models here.

admin.site.site_title = 'CAUC飞行品质监控管理平台'
admin.site.site_header = 'CAUC飞行品质监控管理平台'
admin.site.index_title = 'QAR数据管理'

class QARResource(resources.ModelResource):
    class Meta:
        model = QAR
        # 在导入前添加id列
        def before_import(self, dataset):
            if 'id' not in dataset.headers:
                dataset.headers.append('id')

@admin.register(QAR)
class QARAdmin(ImportExportModelAdmin):
    resource_class = QARResource
    list_display = [field.name for field in QAR._meta.fields[1:]]