from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.views.generic import ListView
import hashlib

from .forms import VehicleForm, DealerRegistrationForm
from .models import (
    Vehicle, Brand, VehicleModel, Trim, State, Town, DealerProfile, InnerColor, SocialMedia, Category, VAS,
    Carousel,
)

class HomeView(ListView):
    """
    Displays the home page with a list of vehicles.
    
    This view also handles filtering based on GET request parameters
    from the filter form.
    """
    model = Vehicle
    template_name = 'myapp/home.html'
    context_object_name = 'vehicles'
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Get filter parameters from the request
        category_id = self.request.GET.get('category')
        brand_id = self.request.GET.get('brand')
        model_id = self.request.GET.get('model')
        trim_id = self.request.GET.get('trim')
        state_id = self.request.GET.get('state')
        town_id = self.request.GET.get('town')
        year_min = self.request.GET.get('year_min')
        year_max = self.request.GET.get('year_max')
        price_min = self.request.GET.get('price_min')
        price_max = self.request.GET.get('price_max')
        
        # Build a Q object to filter the queryset
        q_filters = Q()
        
        if category_id:
            q_filters &= Q(category_id=category_id)
        if brand_id:
            q_filters &= Q(brand_id=brand_id)
        if model_id:
            q_filters &= Q(vehicle_model_id=model_id)
        if trim_id:
            q_filters &= Q(trim_id=trim_id)
        if state_id:
            q_filters &= Q(state_id=state_id)
        if town_id:
            q_filters &= Q(town_id=town_id)
            
        try:
            if year_min:
                q_filters &= Q(manufacture_year__year__gte=int(year_min))
            if year_max:
                q_filters &= Q(manufacture_year__year__lte=int(year_max))
        except (ValueError, TypeError):
            pass # Ignore invalid year inputs
            
        try:
            if price_min:
                q_filters &= Q(price__gte=int(price_min))
            if price_max:
                q_filters &= Q(price__lte=int(price_max))
        except (ValueError, TypeError):
            pass # Ignore invalid price inputs

        return queryset.filter(q_filters)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Pass data for the filter form
        context['categories'] = Category.objects.all()
        context['brands'] = Brand.objects.all()
        context['models'] = VehicleModel.objects.all()
        context['trims'] = Trim.objects.all()
        context['states'] = State.objects.all()
        context['towns'] = Town.objects.all()
        context['carousels'] = Carousel.objects.all()
        
        # Pass the selected filter values to the template
        context['selected_brand'] = self.request.GET.get('brand', '')
        context['selected_model'] = self.request.GET.get('model', '')
        context['selected_trim'] = self.request.GET.get('trim', '')
        context['selected_state'] = self.request.GET.get('state', '')
        context['selected_town'] = self.request.GET.get('town', '')
        context['year_min'] = self.request.GET.get('year_min', '')
        context['year_max'] = self.request.GET.get('year_max', '')
        context['price_min'] = self.request.GET.get('price_min', '')
        context['price_max'] = self.request.GET.get('price_max', '')
        
        return context

@login_required
def upload_vehicle(request):
    """
    Handles the vehicle upload form submission, including image duplicate checks.
    """
    if request.method == 'POST':
        form = VehicleForm(request.POST, request.FILES)
        if form.is_valid():
            image_fields = ['image', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10']
            uploaded_hashes = set()
            
            for field_name in image_fields:
                uploaded_file = request.FILES.get(field_name)
                if uploaded_file:
                    hasher = hashlib.sha256()
                    for chunk in uploaded_file.chunks():
                        hasher.update(chunk)
                    image_hash = hasher.hexdigest()
                    
                    if image_hash in uploaded_hashes:
                        messages.error(request, "Please upload distinct images. A duplicate was found among the uploaded files.")
                        return redirect('dashboard')
                    
                    q_objects = Q()
                    for hash_field in [f'{f}_hash' for f in image_fields]:
                        q_objects |= Q(**{hash_field: image_hash})
                    
                    if Vehicle.objects.filter(q_objects).exists():
                        messages.error(request, "This photo already exists in the database.")
                        return redirect('dashboard')
                    
                    uploaded_hashes.add(image_hash)
            
            vehicle = form.save(commit=False)
            vehicle.seller = request.user
            
            for field_name in image_fields:
                uploaded_file = request.FILES.get(field_name)
                if uploaded_file:
                    hasher = hashlib.sha256()
                    for chunk in uploaded_file.chunks():
                        hasher.update(chunk)
                    image_hash = hasher.hexdigest()
                    setattr(vehicle, f'{field_name}_hash', image_hash)
            
            vehicle.save()
            form.save_m2m()
            
            messages.success(request, "Vehicle uploaded successfully!")
            return redirect('dashboard')
        else:
            context = {
                'form': form,
                'brands': Brand.objects.all(),
                'categories': Category.objects.all(),
                'states': State.objects.all(),
                'trims': Trim.objects.all(),
                'inner_colors': InnerColor.objects.all()
            }
            messages.error(request, "There was an error in your submission. Please check the form and try again.")
            return render(request, 'myapp/upload_vehicle.html', context)
    else:
        form = VehicleForm()
    
    context = {
        'form': form,
        'brands': Brand.objects.all(),
        'categories': Category.objects.all(),
        'states': State.objects.all(),
        'trims': Trim.objects.all(),
        'inner_colors': InnerColor.objects.all()
    }
    return render(request, 'myapp/upload_vehicle.html', context)

@login_required
def dealer_registration(request):
    """
    Handles the dealer registration form submission.
    """
    print("=dddd" * 50)
    # Check if the user is already a dealer
    if DealerProfile.objects.filter(user=request.user).exists():
        messages.info(request, "You are already registered as a dealer.")
        return redirect('dashboard')

    if request.method == 'POST':
        form = DealerRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            dealer_profile = form.save(commit=False)
            dealer_profile.user = request.user
            dealer_profile.save()
            
            messages.success(request, "You have successfully registered as a dealer! You can now list vehicles.")
            return redirect('buyrite:dashboard')
    else:
        form = DealerRegistrationForm()

    context = {
        'form': form,
        'states': State.objects.all(),
    }
    return render(request, 'buyrite/dealer_reg.html', context)
        
def load_models(request):
    """
    HTMX endpoint to load models based on selected brand.
    """
    brand_id = request.GET.get('brand')
    models = VehicleModel.objects.filter(brand_id=brand_id).order_by('name')
    return render(request, 'myapp/partials/_dynamic_models.html', {'models': models, 'selected_model': ''})

def load_trims(request):
    """
    HTMX endpoint to load trims based on selected model.
    """
    model_id = request.GET.get('model')
    trims = Trim.objects.filter(vehicle_model_id=model_id).order_by('name')
    return render(request, 'myapp/partials/_dynamic_trims.html', {'trims': trims, 'selected_trim': ''})

def load_towns(request):
    """
    HTMX endpoint to load towns based on selected state.
    """
    state_id = request.GET.get('state')
    towns = Town.objects.filter(state_id=state_id).order_by('name')
    return render(request, 'myapp/partials/_dynamic_towns.html', {'towns': towns, 'selected_town': ''})

def load_years(request):
    """
    HTMX endpoint to load years.
    """
    # This function is not strictly needed for dynamic filtering,
    # as years are generally static. It's included to match the
    # URL pattern.
    return render(request, 'myapp/partials/_dynamic_years.html')
