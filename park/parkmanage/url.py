from django.urls import path,re_path
from . import views

urlpatterns=[
    path('main/',views.hello),
    path('caradd/',views.car_add),
    path('caraccount/',views.car_account),
    path('finish/',views.finish),
    path('getcnum/',views.getCarNum),
]