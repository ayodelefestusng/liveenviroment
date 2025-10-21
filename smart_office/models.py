from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from taggit.managers import TaggableManager

from django.conf import settings

from django.db import models
from django.db import models
from tinymce.models import HTMLField

from tinymce.models import HTMLField
from django.contrib.auth import get_user_model
User = get_user_model()


class TestPage(models.Model):
    title = models.CharField(max_length=100)
    content = HTMLField()



# class Category(models.Model):
#     name = models.CharField(max_length=100)

#     def __str__(self):
#         return self.name



STATUS_CHOICES = (
    ('draft', 'Draft'),
    ('published', 'Published'),
)


    
    

class Article(models.Model):
    category = models.ForeignKey("Category", on_delete=models.CASCADE)

    title = models.CharField(max_length=200)
    
    content = HTMLField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    tags = TaggableManager()


    def __str__(self):
        return self.title

class Document1(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    submitted_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='submitted_documents')
    
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'),
        ('in_review', 'In Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected')
    ], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)



class Article1(models.Model):
    category = models.ForeignKey("Category", on_delete=models.CASCADE)

    title = models.CharField(max_length=200)
    
    content = HTMLField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    tags = TaggableManager()


    def __str__(self):
        return self.title
    

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver

# Assuming these imports are available in your environment
try:
    from tinymce.models import HTMLField
except ImportError:
    # Fallback if tinymce is not installed, but it's essential for "word-class" content
    HTMLField = models.TextField
    print("WARNING: HTMLField not found. Using models.TextField. Install django-tinymce.")

try:
    from taggit.managers import TaggableManager
except ImportError:
    class TaggableManager(object):
        def __init__(self, *args, **kwargs): pass
    print("WARNING: TaggableManager not found. Install django-taggit.")

# Placeholder for Category model - adjust import as necessary
class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    def __str__(self): return self.name


# --- Global Workflow and Status Choices ---
DOCUMENT_STATUS_CHOICES = (
    ('DRAFT', 'Draft (Author Editing)'),
    ('IN_REVIEW', 'Under Review'),
    ('REVIEW_FEEDBACK', 'Review Feedback Awaiting Author Edit'), # New specific status
    ('IN_CONCURRENCE', 'Pending Concurrence'),
    ('IN_APPROVAL', 'Pending Final Approval'),
    ('APPROVED', 'Approved (Final)'),
    ('REJECTED', 'Rejected / Final Decline'),
)

STAGE_TYPE_CHOICES = (
    ('REVIEW', 'Review'),
    ('CONCURRENCE', 'Concurrence'),
    ('APPROVAL', 'Final Approval'),
)

ASSIGNMENT_STATUS_CHOICES = (
    ('PENDING', 'Pending Action'),
    ('CONCURRED', 'Concurred / Approved'),
    ('REJECTED', 'Rejected / Decline'),
    ('AMENDED', 'Requested Amendment (Concurrence Only)'),
    ('REVIEWED', 'Reviewed / Feedback Provided (Review Only)'),
)


class Document(models.Model):
    """
    The core document object with content, status, and assigned users.
    We track the assigned users for the *initial* workflow setup directly on the document.
    """
    category = models.ForeignKey(Category, on_delete=models.CASCADE, help_text="The category of the document.")
    title = models.CharField(max_length=255)
    
    # Word-Class Content (HTMLField)
    content = HTMLField(help_text="The full, rich-text content of the document.")
    
    file = models.FileField(upload_to='documents/%Y/%m/', blank=True, null=True, help_text="Optional attached file.")
    # submitted_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='submitted_documents')


   # In Document1 model
    # submitted_by = models.ForeignKey(User, on_delete=models.CASCADE)
    submitted_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)


    status = models.CharField(
        max_length=50, 
        choices=DOCUMENT_STATUS_CHOICES, 
        default='DRAFT'
    )
    
    # TaggableManager for document categorization/search
    tags = TaggableManager()
    
    # Fields to store the users selected during submission (M2M relations)
    assigned_reviewers = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='documents_to_review', blank=True)
    assigned_concurrers = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='documents_to_concur', blank=True)
    assigned_approvers = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='documents_to_approve', blank=True)
    
    # Tracks the sequential order of the *current* stage being processed
    current_stage_order = models.IntegerField(default=0) 

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"

    def track_history(self, actor, action_description):
        """Helper to create a historical record."""
        HistoricalRecord.objects.create(
            document=self,
            actor=actor,
            action=action_description
        )

    def is_editable(self):
        """Only the author can edit if status is DRAFT or after receiving feedback/rejection."""
        return self.status in ['DRAFT', 'REVIEW_FEEDBACK', 'REJECTED']
    

    @property
    def initial_reviewers(self):
        return ", ".join(user.email for user in self.assigned_reviewers.all())

    @property
    def initial_concurrers(self):
        return ", ".join(user.email for user in self.assigned_concurrers.all())

    @property
    def initial_approvers(self):
        return ", ".join(user.email for user in self.assigned_approvers.all())


class WorkflowStage(models.Model):
    """
    Defines a required stage (Review, Concurrence, or Approval) in the document's life cycle.
    Created dynamically upon document submission based on the assigned M2M fields on Document.
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='workflow_stages')
    
    # Sequence order (e.g., 1=Review, 2=Concurrence, 3=Approval)
    order = models.IntegerField()
    stage_type = models.CharField(max_length=50, choices=STAGE_TYPE_CHOICES)
    
    is_active = models.BooleanField(default=False) 

    class Meta:
        ordering = ['order']
        unique_together = ('document', 'order')
        verbose_name_plural = "Workflow Stages"

    def __str__(self):
        return f"{self.document.title} - {self.get_stage_type_display()} (Order: {self.order})"


    @property
    def status(self):
        # Aggregate status from related assignments
        statuses = self.assignments.values_list('status', flat=True)
        return ", ".join(set(statuses)) if statuses else "N/A"

    @property
    def completed_at(self):
        # Return latest completion time among assignments
        times = self.assignments.values_list('completed_at', flat=True)
        return max(times) if times else None


class UserAssignment(models.Model):
    """
    Tracks a single user's required action for a specific stage.
    This replaces the separate Reviewer/Approval/Concurrence models for a unified approach.
    """
    stage = models.ForeignKey(WorkflowStage, on_delete=models.CASCADE, related_name='assignments')
    assigned_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='document_assignments')
    
    status = models.CharField(
        max_length=50, 
        choices=ASSIGNMENT_STATUS_CHOICES, 
        default='PENDING'
    )
    
    # The reviewer's feedback/comments (Crucial for all stages)
    comments = models.TextField(blank=True, null=True) 
    
    completed_at = models.DateTimeField(blank=True, null=True)
    @property
    def reviewer(self):
        return self.assigned_user.email

    class Meta:
        # A user should only be assigned once per stage
        unique_together = ('stage', 'assigned_user')
        verbose_name_plural = "User Assignments"

    def __str__(self):
        return f"{self.stage.document.title} - {self.assigned_user.email} ({self.get_status_display()})"
    

class HistoricalRecord(models.Model):
    """Tracks every significant action (submission, edit, review action, status change)."""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='history')
    timestamp = models.DateTimeField(auto_now_add=True)
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, help_text="User who performed the action.")
    action = models.TextField() # Detailed description of the action
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = "Historical Records"

    def __str__(self):
        actor_name = self.actor.email if self.actor else 'System'
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] by {actor_name}: {self.action}"


class Notification(models.Model):
    """Model to track internal notifications for users."""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications')
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    message = models.CharField(max_length=255)
    read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Notification for {self.user.email} on {self.document.title}"


# --- Signal for Notification/History on Assignment Action ---
@receiver(post_save, sender=UserAssignment)
def handle_assignment_completion(sender, instance, created, **kwargs):
    """
    Triggers history tracking and is the place to insert logic for:
    1. Notifying the next person in the sequence.
    2. Checking if all assignments in the current stage are complete.
    3. Transitioning the Document to the next stage or back to the author.
    """
    # Only run logic when an existing assignment changes status from PENDING
    if not created and instance.status != 'PENDING':
        document = instance.stage.document
        
        # 1. Track History
        action_desc = f"{instance.stage.get_stage_type_display()} action by {instance.assigned_user.email}: {instance.get_status_display()}."
        if instance.comments:
            action_desc += f" Comments: {instance.comments[:50]}..."
        document.track_history(instance.assigned_user, action_desc)

        # 2. Add Notification for the document author (or whoever needs to know)
        # This is placeholder logic; full notification service needed in a separate layer
        Notification.objects.create(
            user=document.submitted_by,
            document=document,
            message=f"A {instance.stage.get_stage_type_display()} action was taken on your document: {document.title}."
        )

        # 3. CRITICAL: Trigger the workflow service to check for stage completion and advancement.
        # This complex logic is best handled in a separate service function (e.g., process_workflow_step(document)) 
        # to keep the signal clean and atomic.
        pass
