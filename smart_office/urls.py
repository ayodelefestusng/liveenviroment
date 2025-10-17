from django.urls import path
from . import views

app_name = 'smart_office'

urlpatterns = [
    # Document Management
    path('list/', views.document_list, name='document_list'),
    path('submit/', views.submit_document, name='submit_document'),
    path('<int:pk>/', views.document_detail, name='document_detail'),

    # CORE WORKFLOW ENDPOINT
    # This unified endpoint handles Review, Concurrence, and Approval actions.
    # It requires the primary key of the UserAssignment object.
    path('action/<int:assignment_pk>/', views.document_action, name='document_action'),

    # Auxiliary Features
    # Used by the rich text editor (TinyMCE/CKEditor) for image uploads
    path('upload-image/', views.upload_image, name='upload_image'),
    path('search/', views.search_document, name='search_document'),
]
