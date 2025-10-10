from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("admin/", admin.site.urls),
    path("bert_classifier/", include("bert_classifier.urls")),
    path("dialog_bot/", include("dialog_bot.urls")),
    path("image_classification/", include("image_classification.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
