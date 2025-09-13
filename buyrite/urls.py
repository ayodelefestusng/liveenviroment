from django.urls import path
from .views import (
    HomeView, VehicleDetailView,
    upload_vehicle, upload_vehicle_success,
    load_models, load_trims, load_years, load_towns,
    #  get_models_by_brand, get_trims_by_model,
    DashboardView, mark_as_sold, edit_vehicle,
    dealer_registration, operations_view, approve_dealer,
    reject_dealer, handle_admin_tool_form,
    VehicleImageView, VINImageSearchView, VINImageDrive
)

urlpatterns = [
    # Home & Vehicle
    path('', HomeView.as_view(), name='home'),
    path('vehicle_detail/<slug:slug>/', VehicleDetailView.as_view(), name='vehicle_detail'),

  # Dashboard & Vehicle Actions
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('vehicle/<int:pk>/mark-as-sold/', mark_as_sold, name='mark_as_sold'),
    path('vehicle/<int:pk>/edit/', edit_vehicle, name='edit_vehicle'),

    # Upload Operations
    path('upload/', upload_vehicle, name='upload_vehicle'),
    path('upload/success/', upload_vehicle_success, name='upload_vehicle_success'),

    # HTMX Endpoints
    path('load-models/', load_models, name='load_models'),
    path('load-trims/', load_trims, name='load_trims'),
    path('load-towns/', load_towns, name='load_towns'),
    path('load-years/', load_years, name='load_years'),
    # Load years html yet to be created
#   'buyrite/partials/_vehicle_card.html'

    # path('get-models-by-brand/<int:brand_id>/', get_models_by_brand, name='get_models_by_brand'),
    # path('get-trims/<int:model_id>/', get_trims_by_model, name='get_trims_by_model'),
    # path('get-towns-by-state/<int:state_id>/', get_towns_by_state, name='get_towns_by_state'),

  
    # Dealer Registration & Admin Tools
    path('dealer/register/', dealer_registration, name='dealer_registration'),
    path('operations/', operations_view, name='operations'),
    path('operations/approve/<int:pk>/', approve_dealer, name='approve_dealer'),
    path('operations/reject/<int:pk>/', reject_dealer, name='reject_dealer'),
    path('operations/admin-tool/<str:model_name>/', handle_admin_tool_form, name='handle_admin_tool_form'),

    # API Endpoints
    path('get-vehicle-image/', VehicleImageView.as_view(), name='get_vehicle_image'),
    path('api/vin-search/', VINImageSearchView.as_view(), name='vin_search'),
    path('api2/<str:vin>/', VINImageDrive, name='vin_drive'),
]