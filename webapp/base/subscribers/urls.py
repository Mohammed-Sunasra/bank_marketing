from django.conf.urls import url, include
from django.contrib import admin

from . import views

urlpatterns = [
    url(r'^dashboard/$', views.dashboard_view, name='dashboard')
]
