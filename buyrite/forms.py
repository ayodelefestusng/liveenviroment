from django import forms
from django.apps import apps
from django.contrib.auth import get_user_model

from .models import (
    Post, Tag,
    Vehicle, DealerProfile, InnerColor,
    Category, Brand, VehicleModel, Trim,
    ManufactureYear, FuelOption, Color,
    EngineType, DriveTerrain, Vas,
    State, Town, Condition
)

User = get_user_model()


class PostForm(forms.ModelForm):
    tags = forms.ModelMultipleChoiceField(
        queryset=Tag.objects.all(),
        widget=forms.SelectMultiple
    )

    class Meta:
        model = Post
        fields = ['title', 'content', 'tags']


class VehicleForm(forms.ModelForm):
    # Initial empty querysets for dynamic fields
    vehicle_model = forms.ModelChoiceField(
        queryset=VehicleModel.objects.none(),
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    trim = forms.ModelChoiceField(
        queryset=Trim.objects.none(),
        required=False, # Set to False
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    town = forms.ModelChoiceField(
        queryset=Town.objects.none(),
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    class Meta:
        model = Vehicle
        exclude = ['seller', 'slug', 'index', 'created_at', 'date_sold', 'number_of_view', 'vin']
        widgets = {
            'brand': forms.Select(attrs={'class': 'form-select'}),
            'category': forms.Select(attrs={'class': 'form-select'}),
            'manufacture_year': forms.Select(attrs={'class': 'form-select'}),
            'condition': forms.Select(attrs={'class': 'form-select'}),
            'fuel_option': forms.Select(attrs={'class': 'form-select'}),
            'color': forms.Select(attrs={'class': 'form-select'}),
            'engine_type': forms.Select(attrs={'class': 'form-select'}),
            'drive_terrain': forms.Select(attrs={'class': 'form-select'}),
            'state': forms.Select(attrs={'class': 'form-select'}),
            'exchange_option': forms.Select(attrs={'class': 'form-select'}),
            'registered': forms.Select(attrs={'class': 'form-select'}),
            'negotiable': forms.Select(attrs={'class': 'form-select'}),
            'no_of_tyres': forms.NumberInput(attrs={'class': 'form-control'}),
            'seat': forms.NumberInput(attrs={'class': 'form-control'}),
            'mileage': forms.NumberInput(attrs={'class': 'form-control'}),
            'price': forms.NumberInput(attrs={'class': 'form-control'}),
            'transmission': forms.Select(attrs={'class': 'form-select'}),
            'number_cylinder': forms.NumberInput(attrs={'class': 'form-control'}),
            'horsepower': forms.NumberInput(attrs={'class': 'form-control'}),
            'contact_phone': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'social_media': forms.URLInput(attrs={'class': 'form-control'}),
            'vas': forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input ms-2'}),
            'image': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image2': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image3': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image4': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image5': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image6': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image7': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image8': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image9': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'image10': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'is_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set initial querysets for POST requests
        if 'brand' in self.data:
            try:
                brand_id = int(self.data.get('brand'))
                self.fields['vehicle_model'].queryset = VehicleModel.objects.filter(brand_id=brand_id).order_by('name')
            except (ValueError, TypeError):
                pass
        elif self.instance.pk and self.instance.brand:
            self.fields['vehicle_model'].queryset = VehicleModel.objects.filter(brand=self.instance.brand).order_by('name')

        if 'vehicle_model' in self.data:
            try:
                model_id = int(self.data.get('vehicle_model'))
                self.fields['trim'].queryset = Trim.objects.filter(vehicle_model_id=model_id).order_by('name')
            except (ValueError, TypeError):
                pass
        elif self.instance.pk and self.instance.vehicle_model:
            self.fields['trim'].queryset = Trim.objects.filter(vehicle_model=self.instance.vehicle_model).order_by('name')

        if 'state' in self.data:
            try:
                state_id = int(self.data.get('state'))
                self.fields['town'].queryset = Town.objects.filter(state_id=state_id).order_by('name')
            except (ValueError, TypeError):
                pass
        elif self.instance.pk and self.instance.state:
            self.fields['town'].queryset = Town.objects.filter(state=self.instance.state).order_by('name')

    def clean_trim(self):
        trim = self.cleaned_data.get('trim')
        vehicle_model = self.cleaned_data.get('vehicle_model')

        if vehicle_model:
            # Check if any trims exist for the selected model
            has_trims = Trim.objects.filter(vehicle_model=vehicle_model).exists()
            if has_trims and not trim:
                # If trims exist but none was selected, raise a validation error
                raise forms.ValidationError('This field is required.')

        return trim

class DealerRegistrationForm(forms.ModelForm):
    """
    A form for new dealer registration.
    """
    class Meta:
        model = DealerProfile
        fields = ['business_logo', 'business_address', 'state', 'town']

    def __init__(self, *args, **kwargs):
        # Call the parent's __init__ first
        super().__init__(*args, **kwargs)

        for field_name, field in self.fields.items():
            field.widget.attrs.update({
                'class': 'form-control rounded-lg'
            })

        # Dynamically filter towns based on the submitted state on POST,
        # or the initial state on GET.
        
        # Check if state data is available (either from POST or initial data)
        if 'state' in self.data:
            try:
                state_id = int(self.data.get('state'))
                self.fields['town'].queryset = Town.objects.filter(state_id=state_id).order_by('name')
            except (ValueError, TypeError):
                self.fields['town'].queryset = Town.objects.none()
        elif self.initial.get('state'):
            self.fields['town'].queryset = Town.objects.filter(state=self.initial['state']).order_by('name')
        else:
            self.fields['town'].queryset = Town.objects.none()

class AdminToolForm1(forms.ModelForm):
    """
    A dynamic form to create/edit any specified model.
    
    This form is designed to be reusable. It automatically configures its
    Meta class with the `model` and `fields` based on the model instance
    passed during initialization in the view.
    """
    def __init__(self, model_class, *args, **kwargs):
        # Dynamically set the form's model and fields
        class Meta:
            model = model_class
            fields = '__all__'
        self.Meta = Meta
        super().__init__(*args, **kwargs)
        
        for field_name, field in self.fields.items():
            field.widget.attrs.update({'class': 'form-control rounded-lg'})


from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.decorators import login_required, permission_required
from .models import DealerProfile, State, Town, Category, Brand, VehicleModel, Trim, ManufactureYear, FuelOption, Color, InnerColor, EngineType, DriveTerrain, Vas, Condition
from django import forms
class AdminToolForm(forms.ModelForm):
    """
    A dynamic form to create/edit any specified model.
    
    This form is designed to be reusable. It automatically configures its
    Meta class with the `model` and `fields` based on the model instance
    passed during initialization in the view.
    """
    def __init__(self, model_class, *args, **kwargs):
        # Dynamically set the form's model and fields
        class Meta:
            model = model_class
            fields = '__all__'
        self.Meta = Meta
        super().__init__(*args, **kwargs)
        
        for field_name, field in self.fields.items():
            field.widget.attrs.update({'class': 'form-control rounded-lg'})




class RejectionForm(forms.Form):
    comment = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 4}),
        label="Rejection Comment",
        required=True
    )