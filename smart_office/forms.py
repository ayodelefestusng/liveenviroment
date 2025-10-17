from django import forms
from django.contrib.auth import get_user_model
from django.utils import timezone
from .models import Document, UserAssignment, STAGE_TYPE_CHOICES

User = get_user_model()

# Helper to filter the status choices based on the stage type
def get_assignment_choices(stage_type):
    """Returns status choices relevant to the specific stage type."""
    if stage_type == 'REVIEW':
        # Reviewers can only provide feedback or reject
        return [
            ('REVIEWED', 'Reviewed / Feedback Provided'),
            ('REJECTED', 'Reject and Return to Author'),
        ]
    elif stage_type == 'CONCURRENCE':
        # Concurrers must concur, reject, or request amendment
        return [
            ('CONCURRED', 'Concur (Agree)'),
            ('REJECTED', 'Decline (Reject)'),
            ('AMENDED', 'Request Amendment (Return to Author)'),
        ]
    elif stage_type == 'APPROVAL':
        # Approvers give final approval or rejection
        return [
            # Using CONCURRED status internally for Approval
            ('CONCURRED', 'Approve Document (Final)'), 
            ('REJECTED', 'Decline Final Approval (Reject)'),
        ]
    return []


class DocumentSubmissionForm(forms.ModelForm):
    """
    Form for creating a new Document and assigning all workflow users.
    These fields are temporary and used only for form submission, as the actual 
    M2M fields are on the Document model for initial assignments.
    """
    # Use ModelMultipleChoiceField for user selection
    assigned_reviewers = forms.ModelMultipleChoiceField(
        queryset=User.objects.all().order_by('email'),
        required=False,
        widget=forms.SelectMultiple(attrs={'class': 'select2-users'}),
        label="Reviewers (Step 1)"
    )
    assigned_concurrers = forms.ModelMultipleChoiceField(
        queryset=User.objects.all().order_by('email'),
        required=False,
        widget=forms.SelectMultiple(attrs={'class': 'select2-users'}),
        label="Concurrence Users (Step 2 - All must agree)"
    )
    assigned_approvers = forms.ModelMultipleChoiceField(
        queryset=User.objects.all().order_by('email'),
        required=False,
        widget=forms.SelectMultiple(attrs={'class': 'select2-users'}),
        label="Final Approver(s) (Step 3)"
    )

    class Meta:
        model = Document
        # Include fields for submission
        fields = ('category', 'title', 'content', 'file', 'tags')
        
        # NOTE: The M2M fields for reviewers/concurrers/approvers are handled 
        # as separate fields above and processed manually in views.py.
        
        widgets = {
            'content': forms.Textarea(attrs={'class': 'tinymce-editor'}) # Placeholder for TinyMCE
        }


    def clean(self):
        cleaned_data = super().clean()
        reviewers = cleaned_data.get('assigned_reviewers')
        concurrers = cleaned_data.get('assigned_concurrers')
        approvers = cleaned_data.get('assigned_approvers')

        # Validation: At least one person must be assigned to one of the stages
        if not (reviewers or concurrers or approvers):
            raise forms.ValidationError("You must assign at least one user to the Review, Concurrence, or Approval stage.")
        
        return cleaned_data


class AssignmentActionForm(forms.ModelForm):
    """
    A single unified form used by Reviewers, Concurrers, and Approvers 
    to take action on their assigned task.
    """
    # Override the status field to use the filtered choices dynamically
    status = forms.ChoiceField(
        label="Your Decision",
        choices=[], # Choices are set dynamically in __init__
        widget=forms.RadioSelect
    )
    
    class Meta:
        model = UserAssignment
        # Only the status and comments fields are editable by the action user
        fields = ('status', 'comments')

    def __init__(self, *args, **kwargs):
        # Requires 'stage_type' to be passed (e.g., 'REVIEW', 'CONCURRENCE', 'APPROVAL')
        self.stage_type = kwargs.pop('stage_type', None)
        super().__init__(*args, **kwargs)
        
        if self.stage_type:
            # Dynamically set the available status choices based on the stage
            self.fields['status'].choices = get_assignment_choices(self.stage_type)
        else:
            # Prevent submission if stage_type is missing
            self.fields['status'].choices = [('', 'Error: Stage type missing')]

        # Apply Bootstrap styling to comment field
        self.fields['comments'].widget.attrs.update({
            'class': 'form-control', 
            'rows': 3,
            'placeholder': 'Enter required comments/feedback here...',
            'required': 'required' # Comments are required for any action
        })

    def save(self, commit=True):
        """Custom save to set the completion timestamp."""
        instance = super().save(commit=False)
        if commit:
            instance.completed_at = timezone.now()
            instance.save()
        return instance
