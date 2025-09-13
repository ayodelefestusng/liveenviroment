# Standard library
import hashlib
import requests

# Django core
from django.shortcuts import render, redirect, get_object_or_404
from django.http import (
    HttpResponse, JsonResponse, HttpResponseRedirect, HttpResponseServerError
)
from django.views import View
from django.views.generic import ListView, DetailView
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.models import Group, Permission
from django.utils.decorators import method_decorator
from django.utils.timezone import now
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.template.loader import render_to_string
from django.urls import reverse
from django.apps import apps
from django.db.models import Q, F
from django.contrib import messages
from django.contrib.auth import get_user_model

# Django REST Framework
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Local apps
from .models import (
    Vehicle, Category, State, Town, Brand, VehicleModel, Trim, DealerProfile,
    InnerColor, ManufactureYear, FuelOption, Color, EngineType, DriveTerrain,
    Vas, Condition, Carousel, Post, 
)
from .forms import (
    VehicleForm, DealerRegistrationForm, AdminToolForm, PostForm
)

# Auth user model
User = get_user_model()




class HomeView(ListView):
    model = Vehicle
    template_name = 'buyrite/home.html'
    context_object_name = 'vehicles'
    
    def get_queryset(self):
        queryset = Vehicle.objects.filter(is_available=True).order_by('-created_at')
        
        category_id = self.request.GET.get('category')
        state_id = self.request.GET.get('state')
        town_id = self.request.GET.get('town')
        brand_id = self.request.GET.get('brand')
        model_id = self.request.GET.get('model')
        trim_id = self.request.GET.get('trim')
        min_year = self.request.GET.get('min_year')
        max_year = self.request.GET.get('max_year')
        min_price = self.request.GET.get('min_price')
        max_price = self.request.GET.get('max_price')
        color = self.request.GET.get('color')
        inner_color = self.request.GET.get('inner_color')

        if category_id:
            queryset = queryset.filter(category_id=category_id)
        if state_id:
            queryset = queryset.filter(state_id=state_id)
        if town_id:
            queryset = queryset.filter(town_id=town_id)
        if brand_id:
            queryset = queryset.filter(brand_id=brand_id)
        if model_id:
            queryset = queryset.filter(vehicle_model_id=model_id)
        if trim_id:
            queryset = queryset.filter(trim_id=trim_id)
        if min_year:
            queryset = queryset.filter(manufacture_year__gte=min_year)
        if max_year:
            queryset = queryset.filter(manufacture_year__lte=max_year)
        if min_price:
            queryset = queryset.filter(price__gte=min_price)
        if max_price:
            queryset = queryset.filter(price__lte=max_price)
        if color:
            queryset = queryset.filter(color=color)
        if inner_color:
            queryset = queryset.filter(inner_color=inner_color)
        
        return queryset.select_related('brand', 'vehicle_model', 'trim')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        context['brands'] = Brand.objects.all()
        context['states'] = State.objects.all()
        context['inner_colors'] = InnerColor.objects.all()
        context['colors'] = Vehicle.objects.values_list('color', flat=True).distinct()
        return context



    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Pass filter options for the search forms
        context['brands'] = Brand.objects.all()
        context['states'] = State.objects.all()
        context['years'] = ManufactureYear.objects.all().order_by('-year')
        context['categories'] = Category.objects.all()

        # Pass static content that's not related to the main vehicle list
        context['carousels'] = Carousel.objects.all()
        # context['categories'] = Category.objects.all()
        context['posts'] = Post.objects.all()

        # Pre-select values for persistent filtering
        context['selected_brand'] = self.request.GET.get('brand')
        context['selected_model'] = self.request.GET.get('model')
        context['selected_trim'] = self.request.GET.get('trim')
        context['selected_state'] = self.request.GET.get('state')
        context['selected_town'] = self.request.GET.get('town')
        context['selected_category'] = self.request.GET.get('category')

        context['price_min'] = self.request.GET.get('price_min')
        context['price_max'] = self.request.GET.get('price_max')
        context['year_min'] = self.request.GET.get('year_min')
        context['year_max'] = self.request.GET.get('year_max')

        # Provide a queryset for the dynamic models/trims
        if context['selected_brand']:
            context['models'] = VehicleModel.objects.filter(brand_id=context['selected_brand'])
        else:
            context['models'] = VehicleModel.objects.none()

        if context['selected_model']:
            context['trims'] = Trim.objects.filter(vehicle_model_id=context['selected_model'])
        else:
            context['trims'] = Trim.objects.none()

        if context['selected_state']:
            context['towns'] = Town.objects.filter(state_id=context['selected_state'])
        else:
            context['towns'] = Town.objects.none()

        # Add 'created_ago' to each vehicle in the paginated list
        for vehicle in context['vehicles']:
            time_diff = now() - vehicle.created_at
            days = time_diff.days
            hours = time_diff.seconds // 3600

            if hours < 1:
                vehicle.created_ago = "Just now"
            elif days == 0:
                vehicle.created_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif days == 1:
                vehicle.created_ago = "1 day ago"
            else:
                vehicle.created_ago = f"{days} days ago"

        return context

class VehicleDetailView(DetailView):
    model = Vehicle
    template_name = 'buyrite/vehicle_detail.html'
    slug_url_kwarg = 'slug'
    # pk_url_kwarg = 'pk' 
    slug_field = 'slug'

    def get_object(self, queryset=None):
        # Use a database-level update for better performance and to prevent race conditions
        obj = super().get_object(queryset)
        # Vehicle.objects.filter(slug=self.kwargs.get('slug')).update(number_of_view=F('number_of_view') + 1)
        
        # Vehicle.objects.filter(pk=obj.pk).update(number_of_view=F('number_of_view') + 1)
        Vehicle.objects.filter(slug=obj.slug).update(number_of_view=F('number_of_view') + 1)


        # Now, retrieve the updated object to pass to the template
        return super().get_object(queryset)


    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        vehicle = self.get_object()

        # Collect all non-empty image fields
        image_fields = [
            vehicle.image, vehicle.image2, vehicle.image3, vehicle.image4, vehicle.image5,
            vehicle.image6, vehicle.image7, vehicle.image8, vehicle.image9, vehicle.image10
        ]
        context['carousel'] = [img for img in image_fields if img]
        context['product_info'] = vehicle
        time_diff = now() - vehicle.created_at
        days = time_diff.days
        hours = time_diff.seconds // 3600

        # Format based on age
        if hours < 1.00001:
            created_ago = f"{hours} hour ago"
        elif days == 0:
            created_ago = f"{hours} hours ago"
        elif days == 1:
            created_ago = "1 day ago"
        else:
            created_ago = f"{days} days ago"


        context['created_ago'] = created_ago
        return context

@method_decorator(login_required, name='dispatch')
class DashboardView(ListView):
    model = Vehicle
    template_name = 'buyrite/dashboard.html'
    context_object_name = 'vehicles'
    
    def get_queryset(self):
        status = self.request.GET.get('status', 'unsold')


        # Retrieve all filter parameters from the request
        status = self.request.GET.get('status', 'unsold')
        category_id = self.request.GET.get('category')
        state_id = self.request.GET.get('state')
        town_id = self.request.GET.get('town')
        brand_id = self.request.GET.get('brand')
        model_id = self.request.GET.get('model')
        trim_id = self.request.GET.get('trim')
        min_year = self.request.GET.get('min_year')
        max_year = self.request.GET.get('max_year')
        min_price = self.request.GET.get('min_price')
        max_price = self.request.GET.get('max_price')

        color = self.request.GET.get('color')
        inner_color = self.request.GET.get('inner_color')

        user = self.request.user
        
        # Superuser can see all vehicles; regular users only see their own
        if user.is_superuser:
            queryset = Vehicle.objects.all().order_by('-created_at')
        else:
            queryset = Vehicle.objects.filter(owner=user).order_by('-created_at')

        # Filter based on the selected status
        if status == 'sold':
            queryset = queryset.filter(is_available=False)
        elif status == 'unsold':
            queryset = queryset.filter(is_available=True)
        # If status is 'all', no additional filter is applied.


         # Apply other filters if present
        if category_id:
            queryset = queryset.filter(category_id=category_id)
        if state_id:
            queryset = queryset.filter(state_id=state_id)
        if town_id:
            queryset = queryset.filter(town_id=town_id)
        if brand_id:
            queryset = queryset.filter(brand_id=brand_id)
        if model_id:
            queryset = queryset.filter(vehicle_model_id=model_id)
        if trim_id:
            queryset = queryset.filter(trim_id=trim_id)
        if min_year:
            queryset = queryset.filter(manufacture_year__gte=min_year)
        if max_year:
            queryset = queryset.filter(manufacture_year__lte=max_year)
        if min_price:
            queryset = queryset.filter(price__gte=min_price)
        if max_price:
            queryset = queryset.filter(price__lte=max_price)
        if color:
            queryset = queryset.filter(color=color)
        if inner_color:
            queryset = queryset.filter(inner_color=inner_color)
        
        
        
        return queryset.select_related('brand', 'vehicle_model', 'trim')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        status = self.request.GET.get('status', 'unsold')
        user = self.request.user

        # Get counts for the icon section
        if user.is_superuser:
            all_vehicles_count = Vehicle.objects.count()
            sold_count = Vehicle.objects.filter(is_available=False).count()
            unsold_count = Vehicle.objects.filter(is_available=True).count()
        else:
            all_vehicles_count = Vehicle.objects.filter(owner=user).count()
            sold_count = Vehicle.objects.filter(owner=user, is_available=False).count()
            unsold_count = Vehicle.objects.filter(owner=user, is_available=True).count()

        context['sold_count'] = sold_count
        context['unsold_count'] = unsold_count
        context['all_vehicles_count'] = all_vehicles_count
        context['current_status'] = status

        # Data for the filter form
        context['categories'] = Category.objects.all()
        context['brands'] = Brand.objects.all()
        context['states'] = State.objects.all()
        context['trims'] = Trim.objects.all()
        context['inner_colors'] = InnerColor.objects.all()
        context['colors'] = Vehicle.objects.values_list('color', flat=True).distinct()

    

         # Get selected values to pre-populate the filter form
        context['selected_state'] = self.request.GET.get('state')
        context['selected_brand'] = self.request.GET.get('brand')
        context['selected_category'] = self.request.GET.get('category')
        context['selected_trim'] = self.request.GET.get('trim')
        context['selected_min_year'] = self.request.GET.get('min_year')
        context['selected_max_year'] = self.request.GET.get('max_year')
        context['selected_min_price'] = self.request.GET.get('min_price')
        context['selected_max_price'] = self.request.GET.get('max_price')
        
        return context


@login_required
def mark_as_sold(request, pk):
    """Marks a vehicle as sold and returns the updated vehicle list."""
    if request.method == 'POST':
        vehicle = get_object_or_404(Vehicle, pk=pk)
        
        # Security check: Only allow the owner or a superuser to mark as sold
        if request.user == vehicle.seller or request.user.is_superuser:
            print ("Auo",pk)
            vehicle.is_available = False
            vehicle.save()
            return HttpResponse(status=200) # HTMX expects a success response
        
        return HttpResponse('Unauthorized', status=403)
    return HttpResponse('Invalid Request', status=400)



@login_required
def edit_vehicle(request, pk):
    """Handles vehicle editing via a form in a modal."""
    vehicle = get_object_or_404(Vehicle, pk=pk)
    
    # Security check: Only allow the owner or a superuser to edit
    if request.user != vehicle.seller and not request.user.is_superuser:
        return HttpResponse('Unauthorized', status=403)
        
    if request.method == 'POST':
        form = VehicleForm(request.POST, request.FILES, instance=vehicle)
        if form.is_valid():
            form.save()
            # On successful edit, return a new rendered vehicle card for HTMX swap
            return render(request, 'buyrite/partials/_vehicle_card.html', {'vehicle': vehicle})
        else:
            # If form is invalid, return the form with errors for HTMX to re-render
            return render(request, 'buyrite/partials/_edit_vehicle_modal.html', {'form': form, 'vehicle': vehicle})
    else:
        form = VehicleForm(instance=vehicle)

    return render(request, 'buyrite/partials/_edit_vehicle_modal.html', {'form': form, 'vehicle': vehicle})


@login_required
def upload_vehicle(request):
    """
    Handles the vehicle upload form submission, including image duplicate checks.
    """
    if request.method == 'POST':
        form = VehicleForm(request.POST, request.FILES)
        
        # Debug: Print form data and errors
        print("Form data:", request.POST)
        print("Files:", dict(request.FILES))
        
        if form.is_valid():
            print("Form is valid")
            # List of image fields to check
            image_fields = ['image', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10']
            
            # Store file content and hashes
            file_contents = {}
            image_hashes = {}
            
            # Check for intra-upload duplicates and database duplicates
            for field_name in image_fields:
                uploaded_file = request.FILES.get(field_name)
                if uploaded_file:
                    try:
                        # Read and store file content
                        file_content = b''
                        for chunk in uploaded_file.chunks():
                            file_content += chunk
                        
                        # Calculate hash
                        image_hash = hashlib.sha256(file_content).hexdigest()
                        image_hashes[field_name] = image_hash
                        file_contents[field_name] = file_content
                        
                        # Check for intra-upload duplicates
                        if image_hash in image_hashes.values():
                            # Make sure we're not comparing the same field
                            for other_field, other_hash in image_hashes.items():
                                if other_field != field_name and other_hash == image_hash:
                                    messages.error(request, "Please upload distinct images. A duplicate was found among the uploaded files.")
                                    return redirect('upload_vehicle')
                        
                        # Check database for existing hash
                        q_objects = Q()
                        for hash_field in [f'{f}_hash' for f in image_fields]:
                            q_objects |= Q(**{hash_field: image_hash})
                        
                        if Vehicle.objects.filter(q_objects).exists():
                            messages.error(request, "This photo already exists in the database.")
                            return redirect('upload_vehicle')
                            
                    except Exception as e:
                        messages.error(request, f"Error processing image: {str(e)}")
                        return redirect('upload_vehicle')
            
            # Save the vehicle and its image hashes
            try:
                vehicle = form.save(commit=False)
                vehicle.seller = request.user
                
                # Set the image hash fields
                for field_name, image_hash in image_hashes.items():
                    setattr(vehicle, f'{field_name}_hash', image_hash)
                
                vehicle.save()
                form.save_m2m()
                
                messages.success(request, "Vehicle uploaded successfully!")
                return redirect('dashboard')
                
            except Exception as e:
                messages.error(request, f"Error saving vehicle: {str(e)}")
                return redirect('upload_vehicle')
                
        else:
            print("Form errors:", form.errors)
            messages.error(request, "There was an error in your submission. Please check the form and try again.")
    
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
    return render(request, 'buyrite/upload_vehicle.html', context)


@login_required
def upload_vehicle_success(request):
    return render(request, 'buyrite/upload_success.html')

# my_app/views.py
def load_models(request):
    brand_id = request.GET.get('brand')
    models = VehicleModel.objects.filter(brand_id=brand_id).order_by('name')
    return render(request, 'buyrite/partials/vehicle_model_dropdown.html', {'models': models})


def load_trims(request):
    print("=" * 50)
    print("LOAD_TRIMS VIEW CALLED")
    print("GET parameters:", dict(request.GET))
    print("POST parameters:", dict(request.POST))
    print("=" * 50)
    
    model_id = request.GET.get('vehicle_model')
    print(f"Model ID received: '{model_id}'")
    
    if not model_id:
        print("No model_id provided")
        return HttpResponse('<option value="">Select a model first</option>')
    
    try:
        trims = Trim.objects.filter(vehicle_model_id=model_id).order_by('name')
        print(f"Found {trims.count()} trims for model {model_id}")
        
        return render(request, 'buyrite/partials/trim_dropdown.html', {'trims': trims})
        
    except Exception as e:
        print(f"Error in load_trims: {e}")
        return HttpResponse('<option value="">Error loading trims</option>')

def load_towns(request):
    state_id = request.GET.get('state')
    if not state_id:
        return HttpResponse('<option value="">Select a state first</option>')
    
    towns = Town.objects.filter(state_id=state_id).order_by('name')
    return render(request, 'buyrite/partials/town_dropdown.html', {'towns': towns})



def get_models_by_brandv101(request, brand_id):
    """Returns a partial HTML for models based on the selected brand."""
    models = VehicleModel.objects.filter(brand_id=brand_id).order_by('name')
    return render(request, 'buyrite/partials/_dynamic_models.html', {'models': models})

def load_years(request):
    year_type = request.GET.get('type')
    years = ManufactureYear.objects.all().order_by('-year')
    return render(request, 'buyrite/partials/year_dropdown.html', {'years': years, 'year_type': year_type})





@login_required
def dealer_registration(request):
    """
    Handles the dealer registration form submission.
    """
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
            return redirect('dashboard')
    else:
        form = DealerRegistrationForm()

    context = {
        'form': form,
        'states': State.objects.all(),
    }
    return render(request, 'buyrite/dealer_reg.html', context)


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
        ('VAS', Vas),
        ('State', State),
        ('Town', Town),
        ('Condition', Condition),
    ]
    
    user_permissions = request.user.get_all_permissions()
    allowed_models = []
    
    for name, model in models_to_manage:
        # Check for change/add permissions on the model
        has_permission = any(
            f'buyrite.change_{model.__name__.lower()}' in user_permissions or
            f'buyrite.add_{model.__name__.lower()}' in user_permissions
        )
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
@permission_required('buyrite.change_dealerprofile', raise_exception=True)
def approve_dealer(request, pk):
    """Approves a dealer and sets their is_seller status to True."""
    if request.method == 'POST':
        try:
            dealer_profile = DealerProfile.objects.get(pk=pk)
            dealer_profile.is_confirmed = True
            dealer_profile.user.is_seller = True
            dealer_profile.user.save()
            dealer_profile.save()
            messages.success(request, f"Dealer {dealer_profile.user.username} has been approved.")
            return HttpResponseRedirect(reverse('operations'))
        except DealerProfile.DoesNotExist:
            messages.error(request, "Dealer profile not found.")
            return HttpResponseRedirect(reverse('operations'))
    return HttpResponse('Invalid request', status=400)

@login_required
@permission_required('buyrite.change_dealerprofile', raise_exception=True)
def reject_dealer(request, pk):
    """Rejects a dealer's registration request."""
    if request.method == 'POST':
        try:
            dealer_profile = DealerProfile.objects.get(pk=pk)
            dealer_profile.is_rejected = True
            dealer_profile.save()
            messages.info(request, f"Dealer {dealer_profile.user.username}'s request has been rejected.")
            return HttpResponseRedirect(reverse('operations'))
        except DealerProfile.DoesNotExist:
            messages.error(request, "Dealer profile not found.")
            return HttpResponseRedirect(reverse('operations'))
    return HttpResponse('Invalid request', status=400)

@login_required
def handle_admin_tool_form(request, model_name):
    """
    Handles form submission for creating/modifying models.
    """
    model_map = {
        'Category': Category,
        'Brand': Brand,
        'VehicleModel': VehicleModel,
        'Trim': Trim,
        'ManufactureYear': ManufactureYear,
        'FuelOption': FuelOption,
        'Color': Color,
        'InnerColor': InnerColor,
        'EngineType': EngineType,
        'DriveTerrain': DriveTerrain,
        'VAS': Vas,
        'State': State,
        'Town': Town,
        'Condition': Condition,
    }

    model = model_map.get(model_name)
    if not model:
        return HttpResponse("Model not found.", status=404)

    if not request.user.has_perm(f'buyrite.add_{model.__name__.lower()}') and \
       not request.user.has_perm(f'buyrite.change_{model.__name__.lower()}'):
        return HttpResponse('You do not have permission to perform this action.', status=403)

    if request.method == 'POST':
        form = AdminToolForm(model, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, f"{model_name} added/updated successfully.")
            return HttpResponseRedirect(reverse('operations'))
        else:
            html = render_to_string('buyrite/partials/_admin_tools_forms.html', {'form': form, 'model_name': model_name}, request=request)
            return HttpResponse(html)
    
    form = AdminToolForm(model)
    html = render_to_string('buyrite/partials/_admin_tools_forms.html', {'form': form, 'model_name': model_name}, request=request)
    return HttpResponse(html)




def fetch_vehicle_image(vin, image_size):
    url = "https://zylalabs.com/api/9168/vin+image+capture+for+vehicles+api/16576/get+image"
    params = {
        "vin": vin,
        "image size": image_size
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()  # or response.content if it's an image
    else:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    



class VehicleImageView(View):
    def get(self, request):
        vin = request.GET.get("vin")
        image_size = request.GET.get("image_size")

        if not vin or not image_size:
            return JsonResponse({"error": "Missing vin or image_size"}, status=400)

        try:
            image_data = fetch_vehicle_image(vin, image_size)
            return JsonResponse(image_data)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
        
# views.py

class VINImageSearchView(APIView):
    def get(self, request):
        vin = request.query_params.get("vin")
        image_size = 300  # Hardcoded default

        if not vin:
            return Response({"error": "VIN is required."}, status=status.HTTP_400_BAD_REQUEST)

        # External API call
        url = "https://zylalabs.com/api/9168/vin+image+capture+for+vehicles+api/16576/get+image"
        params = {
            "vin": vin,
            "image size": image_size
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract and format response
            formatted = {
                "vin": vin,
                "image_url": data.get("image_url"),  # Adjust key based on actual response
                "details": data.get("text", "No details provided")  # Adjust key as needed
            }

            return Response(formatted, status=status.HTTP_200_OK)

        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
        


def VINImageDrive(request, vin):
    image_size = 300

    if not vin:
        return JsonResponse({"error": "VIN is required."}, status=400)

    url = "https://zylalabs.com/api/9168/vin+image+capture+for+vehicles+api/16576/get+image"
    params = {
        "vin": vin,
        "image size": image_size
    }


    headers = {
        "Authorization": "Bearer YOUR_API_KEY"
    }


    try:
        # response = requests.get(url, params=params)
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        formatted = {
            "vin": vin,
            "image_url": data.get("image_url"),
            "details": data.get("text", "No details provided")
        }

        return JsonResponse(formatted, status=200)

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=502)