from django.urls import path
from . import views

urlpatterns = [
    path('getrecommendations/<int:entry>', views.modelResponseView, name='recommendation'),
    
]  