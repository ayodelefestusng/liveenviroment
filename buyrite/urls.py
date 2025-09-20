from django.urls import path

from .views import reject_dealer_view  # Updated import to include the new view
from .views import (DashboardView, HomeView, VehicleDetailView,
                    VehicleImageView, VINImageDrive, VINImageSearchView,
                    approve_dealer, dealer_registration, edit_vehicle,
                    handle_admin_tool_form, load_models, load_towns,
                    load_trims, load_years, mark_as_sold, operations_view,
                    upload_vehicle, upload_vehicle_success, yem,check_vin_view)

app_name = 'buyrite'

urlpatterns = [
    # Home & Vehicle Details
    path('', HomeView.as_view(), name='home'),
    path('vehicle_detail/<slug:slug>/', VehicleDetailView.as_view(), name='vehicle_detail'),
    
    # User & Dealer Dashboard
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('dealer/register/', dealer_registration, name='dealer_registration'),

    # Vehicle Management
    path('upload/', upload_vehicle, name='upload_vehicle'),
    path('upload/success/', upload_vehicle_success, name='upload_vehicle_success'),
    path('vehicle/<int:pk>/edit/', edit_vehicle, name='edit_vehicle'),
    path('vehicle/<int:pk>/mark-as-sold/', mark_as_sold, name='mark_as_sold'),

    # Admin & Operations
    path('operations/', operations_view, name='operations'),
    path('operations/approve/<int:pk>/', approve_dealer, name='approve_dealer'),
    # This URL has been updated to use the reject_dealer_view
    path('operations/reject/<int:user_id>/', reject_dealer_view, name='reject_dealer'),
    path('operations/admin-tool/<str:model_name>/', handle_admin_tool_form, name='handle_admin_tool_form'),

    # HTMX Endpoints for Dynamic Forms
    path('load-models/', load_models, name='load_models'),
    path('load-trims/', load_trims, name='load_trims'),
    path('load-towns/', load_towns, name='load_towns'),
    path('load-years/', load_years, name='load_years'),

    # VIN & Image Lookup
    path('api/vin-image-view/', VehicleImageView.as_view(), name='vin_image_view'),
    path('api/vin-image-search/', VINImageSearchView.as_view(), name='vin_image_search'),
    path('api/vin-image-drive/<str:vin>/', VINImageDrive, name='vin_image_drive'),
    
    # Additional Features
    path('yem/', yem, name='yem'),
     path('check-vin/', check_vin_view, name='check_vin'),


]


