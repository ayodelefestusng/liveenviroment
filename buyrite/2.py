from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.decorators import login_required, permission_required
from django.template.loader import render_to_string
from .models import DealerProfile, State, Town, Category, Brand, VehicleModel, Trim, ManufactureYear, FuelOption, Color, InnerColor, EngineType, DriveTerrain, VAS, Condition
from django import forms
from django.apps import apps
from django.conf import settings

# A dictionary mapping model names to their actual classes
MODEL_MAPPING = {
    'category': Category,
    'brand': Brand,
    'vehiclemodel': VehicleModel,
    'trim': Trim,
    'manufactureyear': ManufactureYear,
    'fueloption': FuelOption,
    'color': Color,
    'innercolor': InnerColor,
    'enginetype': EngineType,
    'driveterrain': DriveTerrain,
    'vas': VAS,
    'state': State,
    'town': Town,
    'condition': Condition,
}

# ---
# Form Factory Function to create a dynamic Admin Form
# This is the correct pattern for dynamic ModelForms in Django.
# ---
def create_admin_tool_form(model):
    """
    Creates a dynamic ModelForm class for a given model.
    """
    class AdminToolForm(forms.ModelForm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for field in self.fields.values():
                field.widget.attrs.update({'class': 'form-control rounded-lg'})
    
    # Dynamically set the form's Meta class outside the inner class definition
    # to avoid a NameError with the 'model' variable's scope.
    class Meta:
        model = model
        fields = '__all__'
        
    AdminToolForm.Meta = Meta
    
    return AdminToolForm

@login_required
@permission_required('auth.add_user', raise_exception=True)
def operations_view(request):
    """
    Displays the Operations dashboard for Managers and Superusers.
    
    This view lists dealers who are pending approval, rejected, and approved.
    It also provides access to admin tools based on user permissions.
    """
    unconfirmed_dealers = DealerProfile.objects.filter(is_confirmed=False)
    rejected_dealers = DealerProfile.objects.filter(is_confirmed=False, is_rejected=True)
    approved_dealers = DealerProfile.objects.filter(is_confirmed=True)

    models_to_manage = [
        ('Category', Category),
        ('Brand', Brand),
        ('VehicleModel', VehicleModel),
        ('Trim', Trim),
        ('ManufactureYear', ManufactureYear),
        ('FuelOption', FuelOption),
        ('Color', Color),
        ('InnerColor', InnerColor),
        ('EngineType', EngineType),
        ('DriveTerrain', DriveTerrain),
        ('VAS', VAS),
        ('State', State),
        ('Town', Town),
        ('Condition', Condition),
    ]
    
    user_permissions = request.user.get_all_permissions()
    allowed_models = []
    
    for name, model in models_to_manage:
        # Check for change/add permissions on the model
        has_permission = any((
            f'buyrite.change_{model.__name__.lower()}' in user_permissions,
            f'buyrite.add_{model.__name__.lower()}' in user_permissions
        ))
        if has_permission:
            allowed_models.append({'name': name, 'model_name': model.__name__})
    
    context = {
        'unconfirmed_dealers': unconfirmed_dealers,
        'rejected_dealers': rejected_dealers,
        'approved_dealers': approved_dealers,
        'allowed_models': allowed_models,
    }
    return render(request, 'buyrite/operations.html', context)


@login_required
@permission_required('auth.add_user', raise_exception=True)
def handle_admin_tool_form(request, model_name, pk=None):
    """
    A view to handle the creation and editing of various models dynamically.
    """
    model_class = MODEL_MAPPING.get(model_name.lower())
    
    if not model_class:
        messages.error(request, "Invalid model specified.")
        return HttpResponseRedirect(reverse('operations'))

    instance = None
    if pk:
        instance = get_object_or_404(model_class, pk=pk)
    
    # Use the form factory function to create a dynamic form
    AdminForm = create_admin_tool_form(model_class)

    if request.method == 'POST':
        form = AdminForm(request.POST, request.FILES, instance=instance)
        if form.is_valid():
            form.save()
            messages.success(request, f"{model_name} saved successfully!")
            return HttpResponseRedirect(reverse('operations'))
        else:
            # Re-render the form with errors
            return render(request, 'buyrite/admin_tool_form.html', {'form': form, 'model_name': model_name})
    else:
        form = AdminForm(instance=instance)

    context = {
        'form': form,
        'model_name': model_name.capitalize(),
        'instance': instance,
    }
    return render(request, 'buyrite/admin_tool_form.html', context)
