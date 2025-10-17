from django.contrib import admin
from django.utils.html import format_html
from .models import (
    Document, WorkflowStage, UserAssignment, 
    HistoricalRecord, Notification, Category
)
from tinymce.widgets import TinyMCE
from django.db import models


# --- Inlines for Document Details ---

class UserAssignmentInline(admin.TabularInline):
    """Inline to show individual reviewer/concurrence/approval actions."""
    model = UserAssignment
    extra = 0
    fields = ('reviewer', 'status', 'comments', 'completed_at')
    readonly_fields = ('reviewer', 'completed_at', 'status', 'comments')
    
    def has_add_permission(self, request, obj=None):
        # Prevent manual addition of assignments in the admin. They are created automatically.
        return False
    
    def has_delete_permission(self, request, obj=None):
        # Prevent manual deletion to maintain history integrity.
        return False

class WorkflowStageInline(admin.TabularInline):
    """Inline to show the sequence of stages (Review, Concurrence, Approval)."""
    model = WorkflowStage
    extra = 0
    fields = ('order', 'stage_type', 'status', 'completed_at')
    readonly_fields = ('order', 'stage_type', 'status', 'completed_at')
    show_change_link = True # Allows clicking through to the stage detail
    
    def get_inlines(self, request, obj=None):
        # Nest the assignments under the stage
        return (UserAssignmentInline,)
    
    # Custom formfield to render the rich-text field for better admin display (optional)
    formfield_overrides = {
        models.TextField: {'widget': TinyMCE()}
    }
    
    def has_add_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

class DocumentHistoryInline(admin.TabularInline):
    """Inline to show the audit log for the document."""
    model = HistoricalRecord
    extra = 0
    fields = ('timestamp', 'actor', 'action')
    readonly_fields = ('timestamp', 'actor', 'action')
    
    def has_add_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


# --- Main Admin Classes ---

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    # Fields to display in the main Document change list
    list_display = (
        'title', 
        'submitted_by', 
        'category', 
        'get_current_status', 
        'get_current_stage', # This method caused the error
        'created_at', 
        'updated_at'
    )
    
    # Filters for the change list sidebar
    list_filter = ('status', 'category', 'submitted_by', 'created_at')
    
    # Fields to search across
    search_fields = ('title', 'content', 'submitted_by__username')
    
    # Read-only fields in the detail view
    readonly_fields = ('submitted_by', 'status', 'current_stage_order', 'created_at', 'updated_at')
    
    # Field sets for a better organized detail view
    fieldsets = (
        (None, {
            'fields': ('title', 'category', 'content', 'file', 'tags')
        }),
        ('Workflow Status', {
            'fields': ('status', 'current_stage_order', 'submitted_by'),
            'classes': ('collapse',),
        }),
        ('Initial Assignments', {
            'fields': ('assigned_reviewers', 'assigned_concurrers', 'assigned_approvers'),
            'description': "These fields store the initial people assigned to the workflow. The actual workflow stages are below.",
            'classes': ('collapse',),
        }),
    )
    
    # Inlines for stages
    inlines = [WorkflowStageInline]
    
    # Methods for list_display
    
    def get_current_status(self, obj):
        """Displays the Document status with its display name."""
        # This uses the Document's built-in get_status_display
        return obj.get_status_display()
    get_current_status.short_description = 'Status'
    
    def get_current_stage(self, obj):
        """
        Displays the currently active workflow stage name or the final status.
        
        FIX: WorkflowStage uses get_stage_type_display() because its field is 'stage_type'.
        """
        active_stage = obj.workflow_stages.filter(is_active=True).first()
        if active_stage:
            # CORRECTED: Use get_stage_type_display()
            return active_stage.get_stage_type_display()
        # If no active stage, the document status (e.g., DRAFT, APPROVED, REJECTED) is shown.
        return obj.get_status_display() 
    get_current_stage.short_description = 'Active Stage'


@admin.register(WorkflowStage)
class WorkflowStageAdmin(admin.ModelAdmin):
    list_display = ('document', 'order', 'stage_type', 'status', 'is_active', 'completed_at')
    list_filter = ('stage_type', 'is_active')
    search_fields = ('document__title',)
    inlines = [UserAssignmentInline]
    readonly_fields = ('document', 'order', 'stage_type', 'status', 'is_active', 'completed_at')

@admin.register(UserAssignment)
class UserAssignmentAdmin(admin.ModelAdmin):
    list_display = ('document', 'reviewer', 'status', 'stage_type', 'completed_at')
    list_filter = ('status', 'stage__stage_type')
    search_fields = ('reviewer__username', 'document__title')
    readonly_fields = ('document', 'reviewer', 'status', 'comments', 'completed_at')

    def document(self, obj):
        return obj.stage.document
    
    def stage_type(self, obj):
        return obj.stage.get_stage_type_display()
    
    # Exclude add/delete permissions as assignments must be created via the workflow logic
    def has_add_permission(self, request):
        return False
    def has_delete_permission(self, request, obj=None):
        return False

@admin.register(HistoricalRecord)
class DocumentHistoryAdmin(admin.ModelAdmin):
    list_display = ('document', 'timestamp', 'actor', 'action_summary')
    list_filter = ('timestamp', 'actor')
    search_fields = ('document__title', 'action')
    readonly_fields = ('document', 'timestamp', 'actor', 'action')
    
    def action_summary(self, obj):
        return obj.action[:100] + ('...' if len(obj.action) > 100 else '')
    action_summary.short_description = 'Action'
    
    def has_add_permission(self, request):
        return False
    def has_delete_permission(self, request, obj=None):
        return False

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('user', 'document', 'message', 'read', 'created_at')
    list_filter = ('read', 'created_at')
    search_fields = ('user__username', 'document__title', 'message')
    readonly_fields = ('user', 'document', 'message', 'created_at')

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)
