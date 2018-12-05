from django.urls import path

from . import views

urlpatterns = [
    path('agegender', views.AgeGenderView.as_view(), name='agegender'),
    path('emotion', views.EmotionView.as_view(), name='emotion'),
    path('attributes', views.AttributesView.as_view(), name='attributes')


]