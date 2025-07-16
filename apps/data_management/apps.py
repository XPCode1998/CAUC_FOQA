from django.apps import AppConfig


class DataManagementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.data_management"

    def ready(self):
        import apps.data_management.signals