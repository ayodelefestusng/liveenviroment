from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.core.exceptions import PermissionDenied
from django.db.models import Q
from django.conf import settings
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse
from django.utils import timezone
import os
import logging

# We use relative imports since all components are in the 'docs' app
from .forms import DocumentSubmissionForm, AssignmentActionForm
from .models import (
    Document, WorkflowStage, UserAssignment, HistoricalRecord, Notification, Category,
    ASSIGNMENT_STATUS_CHOICES, STAGE_TYPE_CHOICES
)

logger = logging.getLogger(__name__)


# --- CORE WORKFLOW LOGIC ---


@transaction.atomic
def _process_stage_completion(document, active_stage, actor):
    """
    Checks if the active stage is complete and triggers transition to the next stage 
    or returns the document to the author.
    """
    assignments = UserAssignment.objects.filter(stage=active_stage)
    
    # Check if all users have taken action
    if assignments.filter(status='PENDING').exists():
        return # Stage not complete yet

    # --- Stage Completed, Determine Outcome ---
    
    # Rejection/Feedback check
    has_rejections = assignments.filter(status__in=['REJECTED', 'AMENDED']).exists()
    
    if has_rejections:
        # Send back to author for review
        active_stage.is_active = False
        # Stage status is not explicitly needed here, but keeping for consistency
        # active_stage.status = 'REJECTED' 
        active_stage.completed_at = timezone.now()
        active_stage.save()
        
        # Determine status: REVIEW_FEEDBACK (for author editing) or REJECTED (final)
        if active_stage.stage_type == 'APPROVAL':
            document.status = 'REJECTED' # Final status
            document.track_history(actor, "Document rejected during Final Approval stage.")
            Notification.objects.create(user=document.submitted_by, document=document, message=f"Your document '{document.title}' was ultimately REJECTED.")
        else:
            document.status = 'REVIEW_FEEDBACK'
            document.track_history(actor, f"Stage {active_stage.get_stage_type_display()} rejected/amended. Returned to author for editing.")
            Notification.objects.create(user=document.submitted_by, document=document, message=f"Your document '{document.title}' was returned for revisions.")
            
        document.current_stage_order = 0 # Reset order when bounced back to author
        document.save()
        return

    # --- Full Completion (No Rejections) ---
    
    active_stage.is_active = False
    # active_stage.status = 'COMPLETED' # Status is not a field on WorkflowStage model
    active_stage.completed_at = timezone.now()
    active_stage.save()
    
    next_stage = WorkflowStage.objects.filter(document=document, order=active_stage.order + 1).first()

    if next_stage:
        # Advance to next stage
        next_stage.is_active = True
        next_stage.save()
        
        document.current_stage_order = next_stage.order
        # Update status based on the new stage type
        document.status = f"IN_{next_stage.stage_type}"
        document.save()

        document.track_history(actor, f"Stage {active_stage.get_stage_type_display()} completed. Advancing to {next_stage.get_stage_type_display()}.")

        # Notify the new assignees
        for assignment in UserAssignment.objects.filter(stage=next_stage):
            Notification.objects.create(
                user=assignment.assigned_user,
                document=document,
                message=f"Document '{document.title}' is now assigned to you for {next_stage.get_stage_type_display()}."
            )
            
    else:
        # Final Stage Completed (Approval)
        document.status = 'APPROVED'
        document.current_stage_order = 999 # Marks completion
        document.save()
        
        document.track_history(actor, "Document has received final APPROVAL and is complete.")
        Notification.objects.create(user=document.submitted_by, document=document, message=f"Your document '{document.title}' has been officially APPROVED!")

@transaction.atomic
def _initialize_workflow(document, actor, assigned_reviewers, assigned_concurrers, assigned_approvers):
    """Creates the necessary WorkflowStage and UserAssignment objects upon document submission."""
    
    stage_data = {
        'REVIEW': list(assigned_reviewers),
        'CONCURRENCE': list(assigned_concurrers),
        'APPROVAL': list(assigned_approvers),
    }
    
    order = 1
    stages_to_create = [] 
    
    # 1. Create Stages
    for stage_type, users in stage_data.items():
        if users:
            # We only create a stage if users are assigned to it
            # The is_active status is handled in step 3
            stage = WorkflowStage(document=document, order=order, stage_type=stage_type)
            stages_to_create.append(stage)
            order += 1
    
    if not stages_to_create:
        # Should be caught by form validation, but safe to handle here
        document.status = 'DRAFT'
        document.save()
        document.track_history(actor, "Document created but no workflow assignments found.")
        return

    WorkflowStage.objects.bulk_create(stages_to_create)
    
    # Retrieve stages to get IDs
    stages = WorkflowStage.objects.filter(document=document).order_by('order')

    assignments_to_create = []
    
    # 2. Create Assignments (must be done after stages exist)
    for stage in stages:
        user_list = stage_data.get(stage.stage_type)
        for user in user_list:
            assignments_to_create.append(UserAssignment(
                stage=stage, 
                assigned_user=user,
                # REMOVED: The 'action_type' keyword argument which caused the TypeError
                status='PENDING' # Explicitly set status
            ))
    
    UserAssignment.objects.bulk_create(assignments_to_create)
    
    # 3. Activate the first stage
    first_stage = stages.first()
    first_stage.is_active = True
    first_stage.save()
    
    document.status = f"IN_{first_stage.stage_type}" # Start the process
    document.current_stage_order = first_stage.order
    document.save()

    # Notify initial assignees
    for assignment in UserAssignment.objects.filter(stage=first_stage):
        Notification.objects.create(
            user=assignment.assigned_user,
            document=document,
            message=f"New document '{document.title}' assigned to you for {first_stage.get_stage_type_display()}."
        )
        
    document.track_history(actor, f"Workflow initialized and submitted to {first_stage.get_stage_type_display()}.")

# --- VIEWS ---

@login_required
@transaction.atomic
def submit_document(request):
    """Handles document creation and workflow initialization."""
    if request.method == 'POST':
        # Using the actual form class
        form = DocumentSubmissionForm(request.POST, request.FILES) 
        if form.is_valid():
            document = form.save(commit=False)
            document.submitted_by = request.user
            document.status = 'DRAFT' # Initial save as draft before setting workflow status
            document.save()
            
            # Save M2M fields (tags)
            form.save_m2m() 

            # Extract M2M users from the cleaned data
            assigned_reviewers = form.cleaned_data.get('assigned_reviewers')
            assigned_concurrers = form.cleaned_data.get('assigned_concurrers')
            assigned_approvers = form.cleaned_data.get('assigned_approvers')


            # The M2M fields should already be saved via form.save_m2m() 
            # if they were defined on the DocumentSubmissionForm and the model. 
            # We are keeping this manual setting for the document fields 
            # (which store the initial assignments) because they are M2M fields 
            # on the Document model itself.
            document.assigned_reviewers.set(assigned_reviewers) 
            document.assigned_concurrers.set(assigned_concurrers)
            document.assigned_approvers.set(assigned_approvers)
            
            # Initialize the full workflow (Stages and Assignments)
            _initialize_workflow(
                document, request.user, 
                assigned_reviewers, assigned_concurrers, assigned_approvers
            )
            
            messages.success(request, f'Document "{document.title}" submitted and workflow initiated.')
            return redirect('document_detail', pk=document.pk)
    else:
        # Using the actual form class
        form = DocumentSubmissionForm()
        
    return render(request, 'submit_document.html', {'form': form})


@login_required
@transaction.atomic
def edit_document(request, pk):
    """Allows author to edit content when document is DRAFT or REVIEW_FEEDBACK."""
    document = get_object_or_404(Document, pk=pk)
    
    # Permission check: Only author can edit, and only in specific statuses
    if document.submitted_by != request.user:
        messages.error(request, "You do not have permission to edit this document.")
        return redirect('document_detail', pk=pk)

    if not document.is_editable():
        messages.warning(request, f"Document cannot be edited while in status: {document.get_status_display()}.")
        return redirect('document_detail', pk=pk)

    if request.method == 'POST':
        form = DocumentSubmissionForm(request.POST, request.FILES, instance=document)
        if form.is_valid():
            document = form.save(commit=False)
            
            # If the status was REVIEW_FEEDBACK, resubmit and restart workflow
            if document.status == 'REVIEW_FEEDBACK':
                # The document is resubmitted, restart the workflow based on original assignments
                document.status = 'DRAFT' # Temporarily reset
                document.save()
                
                # Clear existing stage/assignment records (or mark them as superseded)
                # For simplicity here, we assume a full reset:
                document.workflow_stages.all().delete()
                
                # NOTE: document.initial_reviewers is not defined in docs/models.py, 
                # so we must use the currently assigned M2M fields (assigned_reviewers, etc.)
                _initialize_workflow(
                    document, request.user, 
                    document.assigned_reviewers.all(), 
                    document.assigned_concurrers.all(), 
                    document.assigned_approvers.all()
                )
                
                messages.success(request, f'Document "{document.title}" updated and resubmitted for review.')
            else:
                # Still in DRAFT, just saving changes
                document.save()
                document.track_history(request.user, "Document content updated while in Draft status.")
                messages.success(request, f'Document "{document.title}" saved successfully.')
                
            return redirect('document_detail', pk=document.pk)
    else:
        form = DocumentSubmissionForm(instance=document)

    return render(request, 'edit_document.html', {'form': form, 'document': document})


@login_required
@transaction.atomic
def document_action(request, assignment_pk):
    """
    Unified view to handle action (Review, Concurrence, Approval) on a single UserAssignment.
    """
    assignment = get_object_or_404(UserAssignment, pk=assignment_pk)
    document = assignment.stage.document
    
    # Security check: Only the assigned user can take action
    if assignment.assigned_user != request.user:
        raise PermissionDenied("You are not authorized to act on this assignment.")
    
    # Prevent re-submitting an already completed assignment
    if assignment.status != 'PENDING':
        messages.error(request, "This assignment has already been completed.")
        return redirect('document_detail', pk=document.pk)

    if request.method == 'POST':
        # Create the form, binding it to the assignment instance
        form = AssignmentActionForm(
            request.POST, 
            instance=assignment, 
            stage_type=assignment.stage.stage_type
        )
        
        if form.is_valid():
            # Save the status and comments (the form handles completed_at)
            completed_assignment = form.save() 
            
            # Add history entry
            document.track_history(
                request.user, 
                f"Action recorded ({completed_assignment.get_status_display()}) in {assignment.stage.get_stage_type_display()} stage."
            )
            
            # Process the stage to check for completion and trigger advancement
            _process_stage_completion(document, assignment.stage, request.user)
            
            messages.success(request, f"Your {assignment.stage.get_stage_type_display()} action has been recorded.")
            return redirect('document_detail', pk=document.pk)
    
    messages.error(request, "Invalid form submission.")
    return redirect('document_detail', pk=document.pk)


@login_required
def document_list(request):
    """Lists all documents, highlighting documents needing user's action."""
    
    # Documents submitted by the user
    my_submitted_documents = Document.objects.filter(submitted_by=request.user).order_by('-updated_at')
    
    # Documents requiring the user's action (Pending assignments in an active stage)
    documents_for_action = UserAssignment.objects.filter(
        assigned_user=request.user,
        status='PENDING',
        stage__is_active=True
    ).select_related('stage__document', 'stage').order_by('stage__document__updated_at')
    
    context = {
        'my_submitted_documents': my_submitted_documents,
        'documents_for_action': documents_for_action,
    }
    return render(request, 'document_list.html', context)





@login_required
def document_detail(request, pk):
    """
    Displays document, history, current status, and the relevant action form 
    if the user has a pending assignment.
    """
    document = get_object_or_404(Document, pk=pk)
    history = HistoricalRecord.objects.filter(document=document).order_by('-timestamp')
    
    # FIX APPLIED HERE: Using the correct related_name 'assignments' instead of 'userassignment_set'
    all_stages = WorkflowStage.objects.filter(document=document).prefetch_related('assignments') 

    user_assignment = None
    action_form = None
    
    # Check if the document is editable by the author
    can_edit_content = document.is_editable() and document.submitted_by == request.user

    # Check for active assignment
    user_assignment = UserAssignment.objects.filter(
        stage__document=document,
        assigned_user=request.user,
        status='PENDING',
        stage__is_active=True
    ).select_related('stage').first()

    if user_assignment:
        # Create the action form dynamically
        action_form = AssignmentActionForm(
            initial={'assignment_id': user_assignment.pk},
            stage_type=user_assignment.stage.stage_type
        )
        # Apply Bootstrap styling to comment field
        action_form.fields['comments'].widget.attrs.update({'class': 'form-control', 'rows': 3})

    context = {
        'document': document,
        'all_stages': all_stages, # For workflow progress visualization
        'history': history,
        'can_edit_content': can_edit_content,
        'user_assignment': user_assignment, # Pending task for the current user
        'action_form': action_form,# Form to take action
        # If the author can edit, we pass the submission form instance for editing fields
        'form': DocumentSubmissionForm(instance=document) if can_edit_content else None, 
    }
    
    return render(request, 'document_detail.html', context)


def get_status_choices():
    """Helper to provide status choices for search filter."""
    # We only show statuses relevant to search:
    statuses = [
        ('APPROVED', 'Approved (Published)'),
        ('REVIEW_FEEDBACK', 'Rejected / Revisions Needed'),
        ('IN_REVIEW', 'In Review'),
        ('IN_CONCURRENCE', 'In Concurrence'),
        ('IN_APPROVAL', 'In Final Approval'),
    ]
    return statuses


@login_required
def search_document(request):
    """Handles document search and filtering based on keywords, category, and status."""
    query = request.GET.get('q', '').strip()
    category_id = request.GET.get('category', '')
    status = request.GET.get('status', '')

    # Start with all documents (for administrators/authenticated view)
    documents = Document.objects.all().order_by('-updated_at')
    
    # Filter by keyword (title or content)
    if query:
        documents = documents.filter(Q(title__icontains=query) | Q(content__icontains=query))
        
    # Filter by category
    if category_id:
        documents = documents.filter(category__id=category_id)
        
    # Filter by status
    if status:
        documents = documents.filter(status=status)

    categories = Category.objects.all()
    status_choices = get_status_choices()
    
    context = {
        'documents': documents,
        'categories': categories,
        'status_choices': status_choices,
        'query': query,
        'selected_category': category_id,
        'selected_status': status,
    }
    
    return render(request, 'search_results.html', context)


@csrf_exempt
def upload_image(request):
    """Handles image uploads from TinyMCE, saving to media folder."""
    if request.method == 'POST':
        image = request.FILES.get('file')
        if not image:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
            
        # Basic security: Ensure it's an image
        if not image.content_type.startswith('image/'):
            return JsonResponse({'error': 'Invalid file type'}, status=400)
            
        # Use os.path.join for path safety and uniqueness
        filename = f"{timezone.now().strftime('%Y%m%d%H%M%S')}_{image.name}"
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'documents', 'uploads')
        
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, filename)
        
        with open(path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
                
        # Return the public URL for TinyMCE
        media_url = f'{settings.MEDIA_URL}documents/uploads/{filename}'
        return JsonResponse({'location': media_url})
        
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def autosave_article(request):
    """Placeholder for autosave functionality (often handled by TinyMCE/JS)."""
    if request.method == 'POST':
        # Logic to save partial content via AJAX
        return HttpResponse("✅ Saved")
    return HttpResponse("Waiting…")
