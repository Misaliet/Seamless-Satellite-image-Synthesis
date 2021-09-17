"""SSS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from sss_ui.views import index
from sss_ui.views import left
from sss_ui.views import up
from sss_ui.views import down
from sss_ui.views import right
from sss_ui.views import transfer
from sss_ui.views import zoom_in
from sss_ui.views import super_resolution
from sss_ui.views import zoom_out
from sss_ui.views import random
from sss_ui.views import restore
from sss_ui.views import get_z

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', index, name='index'),
    path('left/', left, name='left'),
    path('up/', up, name='up'),
    path('down/', down, name='down'),
    path('right/', right, name='right'),
    path('transfer/', transfer, name='transfer'),
    path('zoom_in/', zoom_in, name='zoom_in'),
    path('super_resolution/', super_resolution, name='super_resolution'),
    path('zoom_out/', zoom_out, name='zoom_out'),
    path('random/', random, name='random'),
    path('restore/', restore, name='restore'),
    path('get_z/', get_z, name='get_z'),
]
