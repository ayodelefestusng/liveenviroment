"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information, see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),

    # Core apps
    path('', include(('users.urls', 'users'), namespace='users')),
    path('payments/', include(('crossborder.urls', 'crossborder'), namespace='crossborder')),
    path('ai/', include(('ai.urls', 'ai'), namespace='ai')),
    path('buyrite/', include(('buyrite.urls', 'buyrite'), namespace='buyrite')),
    path('laundry/', include(('laundry.urls', 'laundry'), namespace='laundry')),
    path('store/', include(('store.urls', 'store'), namespace='store')),

    # Authentication
    path('accounts/', include('django.contrib.auth.urls')),  # Built-in login/logout/password views
    path('accounts/', include('allauth.urls')),              # ✅ Social login (Google, Facebook, etc.)
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)