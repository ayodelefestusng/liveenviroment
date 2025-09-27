from django.urls import path
from . import views

app_name = 'store'

urlpatterns = [
    # Home and product views
    path('', views.home, name='home'),
    path('products/<slug:slug>/', views.product_detail, name='product-detail'),
    path('operational/', views.operational_dashboard, name='operational'),

    # Order and Cart management
    path('create-order/', views.create_order, name='create-order'),
    # path('add-to-cart/', views.add_to_cart, name='add-to-cart'),
    path('add-to-cart/', views.add_to_cart, name='add-to-cart'),
    path('remove-from-cart/<int:product_id>/', views.remove_from_cart, name='remove-from-cart'),
    path('edit-cart-item/<int:product_id>/', views.edit_cart_item, name='edit-cart-item'),
    path('get-edit-cart-item-form/<int:product_id>/', views.get_edit_cart_item_form, name='get-edit-cart-item-form'),
    path('view-cart/', views.view_cart, name='view-cart'),
    path('cart-dashboard/<int:pk>/', views.order_dashboard, name='order-dashboard'),
    path('order-list/', views.order_list, name='order-list'),
    
    # Customer and Refund views
    path('create-customer/', views.customer_registration, name='create-customer'),
    path('create-refund/', views.create_refund, name='create-refund'),
    path('refund-order-items/<int:pk>/', views.refund_order_items, name='refund-order-items'),

    # Checkout
    path('checkout/', views.checkout, name='checkout'),
    
    # HTMX
    path('check-customer-by-phone/', views.check_customer_by_phone, name='check-customer-by-phone'),
    path('complete-order/', views.complete_order, name='complete-order'),

    # Operational views
    path('create-product/', views.create_product, name='create-product'),
    path('create-category/', views.create_category, name='create-category'),
    path('create-payment-mode/', views.create_payment_mode, name='create-payment-mode'),


    path('check-credit-mode/<int:mode_id>/', views.check_credit_mode, name='check-credit-mode'),
    path('toggle-credit/', views.toggle_credit, name='toggle-credit'),

    path('refund-order-items/<int:pk>/', views.refund_order_items, name='refund-order-items'),
path('submit-refund/', views.submit_refund, name='submit-refund'),
]


