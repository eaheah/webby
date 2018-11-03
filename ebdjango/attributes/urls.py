from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('test', views.TestView.as_view(), name='model'),
    path('upload', views.simple_upload, name='upload'),


]