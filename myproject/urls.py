"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information, see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView


    # config/urls.py


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

urlpatterns += [
    # path('admin/', admin.site.urls),
    # path('home', include('apps.home.urls')),
    # path('solutions/', include('apps.solutions.urls')),
    # path('platform/', include('apps.platform.urls')),
    # path('industries/', include('apps.industries.urls')),
    # path('about/', include('apps.about.urls')),
    # path('contact/', include('apps.contact.urls')),
    # path('demo/', include('apps.demo.urls')),
    
    # Static pages
    path('privacy-policy/', TemplateView.as_view(template_name='legal/privacy_policy.html'), name='privacy_policy'),
    path('terms-of-service/', TemplateView.as_view(template_name='legal/terms_of_service.html'), name='terms_of_service'),
    path('cookie-policy/', TemplateView.as_view(template_name='legal/cookie_policy.html'), name='cookie_policy'),
]

# Serve media files during development
if settings.DEBUG:

    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


