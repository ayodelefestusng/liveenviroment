from django.db import models

# Create your models here.
# Standard library
import os
import uuid
import hashlib

# Third-party
from PIL import Image

# Django core
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.utils.text import slugify
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

# Django auth
from django.contrib.auth.models import (
    AbstractUser, AbstractBaseUser, BaseUserManager, PermissionsMixin
)
from django.contrib.auth import get_user_model

# Auth user model instance
User = get_user_model()



class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)

class Users(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    full_name = models.CharField(max_length=255, blank=True, null=True)  # ← Add this if missing
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    # Role flags
    is_seller = models.BooleanField(default=False)
    is_buyer = models.BooleanField(default=True)


    # mfa_secret = models.CharField(max_length=16, blank=True, null=True)
    # mfa_enabled = models.BooleanField(default=False)

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='users_users',  # <-- Add this line
        blank=True,
        help_text='The groups this user belongs to.'
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='users_user_permissions',  # <-- Add this line
        blank=True,
        help_text='Specific permissions for this user.'
    )


    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.emai
    username = None
    email = models.EmailField(unique=True)
    full_name = models.CharField(('full name'), max_length=255)
    phone = models.IntegerField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ['full_name']

    objects = CustomUserManager()

    def __str__(self):
        return self.email
    class Meta:
        verbose_name = ('User')
        verbose_name_plural = ('Users')

class Carousel(models.Model):
    name = models.CharField(unique=True,max_length=100)
    image = models.ImageField(upload_to='carosel_images/')

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']  # Orders categories alphabetically by name

class Tag(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    tags = models.ManyToManyField(Tag)


# Extend the default User model with a property to check if they are a dealer
def get_is_dealer(self):
    return hasattr(self, 'dealerprofile')

User.add_to_class('is_dealer', property(get_is_dealer))

# class Customer(AbstractUser):
#     is_seller = models.BooleanField(default=False)
#     is_buyer = models.BooleanField(default=True)
#     phone_number = models.CharField(max_length=20, blank=True)

class Categorys(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name
    
class Category(models.Model):
    name = models.CharField(unique=True,max_length=100)
    image = models.ImageField(upload_to='category_images/')

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']  # Orders categories alphabetically by name

# Represents the manufacturer (e.g., Toyota, BMW).
class Brand(models.Model):
    name = models.CharField(max_length=100, unique=True)
    logo = models.ImageField(upload_to='vehicles/logo')

    def __str__(self):
        return str(self.name)
    class Meta:
        ordering = ['pk']  # Orders categories alphabetically by name

# Linked to a brand (e.g., Corolla under Toyota).
class VehicleModel(models.Model):
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE)
    name = models.CharField(max_length=100,unique=True)

    class Meta:
        unique_together = ('brand', 'name')
        ordering = ['name'] 

    def __str__(self):
        # return f"{self.brand.name} {self.name}"
        return f"{self.pk} {self.name}"
    

# Specific version of a model (e.g., Corolla LE, Corolla XSE)
class Trim(models.Model):
    vehicle_model = models.ForeignKey(VehicleModel, on_delete=models.CASCADE)
    name = models.CharField(max_length=100, default="",blank=True, null=tuple)

    class Meta:
        unique_together = ('vehicle_model', 'name')
        ordering = ['name'] 

    def __str__(self):
        return f"{self.vehicle_model} {self.name}"
    
class ManufactureYear(models.Model):
    year = models.PositiveIntegerField(unique=True)

    def __str__(self):
        return str(self.year)
    class Meta:
        ordering = ['year']  # Orders categories alphabetically by name

class FuelOption(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name
    class Meta:
        ordering = ['name'] 

class Color(models.Model):
    name = models.CharField(max_length=50)
    hex_code = models.CharField(max_length=7, blank=True, null=True)

    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name'] 
class InnerColor(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name
    class Meta:
        ordering = ['name'] 

class EngineType(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name
    class Meta:
        ordering = ['name'] 

class DriveTerrain(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name
    class Meta:
        ordering = ['name'] 

class Condition(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name
    class Meta:
        ordering = ['name'] 

class Vas(models.Model):
    name = models.CharField(max_length=100)

    class Meta:
        verbose_name = "Value Added Service"
        verbose_name_plural = "Value Added Services"

        ordering = ['name']



    def __str__(self):
        return self.name

class State(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name
    class Meta:
        ordering = ['name'] 

class Town(models.Model):
    name = models.CharField(max_length=100)
    state = models.ForeignKey(State, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.name}, {self.state.name}"
    class Meta:
        ordering = ['name'] 
    
BOOLEAN_CHOICES = [
    ('yes', 'Yes'),
    ('no', 'No'),
]
TRANSMISSION_CHOICES = [
    ('automatic', 'Automatic'),
    ('manual', 'Manual'),
]


# Assuming other models (Category, Brand, etc.) are defined above this class
# and BOOLEAN_CHOICES, TRANSMISSION_CHOICES, and related imports are correct.

class Vehicle(models.Model):
    # ForeignKey fields
    seller = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    category = models.ForeignKey('Category', on_delete=models.SET_NULL, null=True)
    brand = models.ForeignKey('Brand', on_delete=models.SET_NULL, null=True)
    vehicle_model = models.ForeignKey('VehicleModel', on_delete=models.SET_NULL, null=True)
    trim = models.ForeignKey('Trim', on_delete=models.SET_NULL, null=True)
    manufacture_year = models.ForeignKey('ManufactureYear', on_delete=models.SET_NULL, null=True)
    condition = models.ForeignKey('Condition', on_delete=models.SET_NULL, null=True)
    fuel_option = models.ForeignKey('FuelOption', on_delete=models.SET_NULL, null=True)
    color = models.ForeignKey('Color', on_delete=models.SET_NULL, null=True)
    inner_color = models.ForeignKey('InnerColor', on_delete=models.SET_NULL, null=True)
    engine_type = models.ForeignKey('EngineType', on_delete=models.SET_NULL, null=True)
    drive_terrain = models.ForeignKey('DriveTerrain', on_delete=models.SET_NULL, null=True)
    state = models.ForeignKey('State', on_delete=models.SET_NULL, null=True)
    town = models.ForeignKey('Town', on_delete=models.SET_NULL, null=True)

    # Boolean and numeric fields
    BOOLEAN_CHOICES = [('yes', 'Yes'), ('no', 'No')]
    TRANSMISSION_CHOICES = [('automatic', 'Automatic'), ('manual', 'Manual')]
    exchange_option = models.CharField(max_length=3, choices=BOOLEAN_CHOICES)
    registered = models.CharField(max_length=3, choices=BOOLEAN_CHOICES)
    negotiable = models.CharField(max_length=3, choices=BOOLEAN_CHOICES)
    no_of_tyres = models.PositiveIntegerField(default=4, validators=[MaxValueValidator(32)])
    seat = models.PositiveIntegerField(default=5, validators=[MaxValueValidator(80)])
    mileage = models.PositiveIntegerField()
    price = models.DecimalField(max_digits=12, decimal_places=2)
    transmission = models.CharField(max_length=10, choices=TRANSMISSION_CHOICES, default='automatic')
    number_cylinder = models.PositiveIntegerField(blank=True, null=True, validators=[MaxValueValidator(20)])
    horsepower = models.PositiveIntegerField(blank=True, null=True, validators=[MaxValueValidator(200)])

    # Text and contact
    contact_phone = models.CharField(max_length=20, blank=True)
    description = models.TextField()
    social_media = models.URLField(blank=True, null=True)

    # Many-to-many
    vas = models.ManyToManyField('Vas')

    # Image fields
    image = models.ImageField(upload_to='vehicles/')
    image2 = models.ImageField(upload_to='vehicles/')
    image3 = models.ImageField(upload_to='vehicles/')
    image4 = models.ImageField(upload_to='vehicles/')
    image5 = models.ImageField(upload_to='vehicles/')
    image6 = models.ImageField(upload_to='vehicles/', blank=True, null=True)
    image7 = models.ImageField(upload_to='vehicles/', blank=True, null=True)
    image8 = models.ImageField(upload_to='vehicles/', blank=True, null=True)
    image9 = models.ImageField(upload_to='vehicles/', blank=True, null=True)
    image10 = models.ImageField(upload_to='vehicles/', blank=True, null=True)

    # Hash fields for image uniqueness validation
    image_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image2_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image3_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image4_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image5_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image6_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image7_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image8_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image9_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    image10_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)

    is_available = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    date_sold = models.DateTimeField(blank=True, null=True)
    number_of_view = models.PositiveIntegerField(default=0)

    # Metadata
    slug = models.SlugField(unique=True, max_length=255, blank=True)
    vin = models.CharField(max_length=50, blank=True)
    index = models.CharField(max_length=130, blank=True, unique=True, db_index=True)

    def clean(self):
        super().clean()
        image_fields = [
            self.image, self.image2, self.image3, self.image4, self.image5,
            self.image6, self.image7, self.image8, self.image9, self.image10
        ]
        
        for image in image_fields:
            if image:
                try:
                    img = Image.open(image.file)
                    width, _ = img.size
                    if not (200 <= width <= 1000):
                        raise ValidationError(
                            f"Image '{image.name}' must be between 200 and 1000 pixels wide. Current width: {width}px."
                        )
                except FileNotFoundError:
                    # This happens when a file is not yet saved to disk (e.g., during tests)
                    pass
                except Exception as e:
                    raise ValidationError(f"Could not validate image '{image.name}': {e}")
    
    def save(self, *args, **kwargs):
        # Determine if this is a new instance
        is_new_instance = self._state.adding

        # Set contact_phone if it's not provided
        if not self.contact_phone and hasattr(self.seller, 'phone'):
            self.contact_phone = self.seller.phone

        # Handle the date_sold logic: set date_sold only when the vehicle is marked as unavailable
        # and it wasn't previously unavailable
        if not self.is_available and self.date_sold is None:
            self.date_sold = timezone.now()
        elif self.is_available:
            self.date_sold = None
        
        # Generate slug and index only when the instance is new or has a change in key fields
        if is_new_instance or self.pk is None:
            # Generate the slug
            base_slug = slugify(f"{self.brand.name}-{self.vehicle_model.name}")
            self.slug = base_slug
            counter = 1
            while Vehicle.objects.filter(slug=self.slug).exists():
                self.slug = f"{base_slug}-{counter}"
                counter += 1

            # Generate the index
            base_index = slugify(f"{self.brand.name}-{self.trim.name}-{self.category.name}-"
                                 f"{self.manufacture_year.year}-{self.vehicle_model.name}-"
                                 f"{self.condition.name}-{self.fuel_option.name}-"
                                 f"{self.engine_type.name}-{self.drive_terrain.name}-"
                                 f"{self.state.name}-{self.town.name}")
            self.index = base_index
            counter = 1
            while Vehicle.objects.filter(index=self.index).exists():
                self.index = f"{base_index}-{counter}"
                counter += 1
        
        # Finally, save the object to the database (only once!)
        super().save(*args, **kwargs)

    # def __str__(self):
        return f"{self.brand.name} {self.vehicle_model.name} {self.trim.name} ({self.manufacture_year.year})"

    class Meta:
        ordering = ['created_at']



class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    excerpt = models.TextField(blank=True, editable=False)  # Auto-generated summary
    slug = models.SlugField(unique=True, max_length=255,blank=True)
    author = models.CharField(max_length=100, default='Anonymous')  # Optional: Add author field
    published_date = models.DateTimeField(default=timezone.now)     # Optional: Add timestamp
    tags = models.ManyToManyField('Tag', blank=True)                # Optional: Add tagging

    def save(self, *args, **kwargs):
        if not self.slug:
            base_slug = slugify(self.title)
            slug = base_slug
            counter = 1
            while Article.objects.filter(slug=slug).exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = slug

                # Auto-generate excerpt if not set
        if not self.excerpt:
            plain_text = self.content.strip().replace('\n', ' ')
            self.excerpt = plain_text[:250] + '...' if len(plain_text) > 250 else plain_text
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title
    



# New: DealerProfile model to store dealer-specific info
class DealerProfile(models.Model):
    # user = models.OneToOneField(User, on_delete=models.CASCADE)
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    business_logo = models.ImageField(upload_to='dealer_logos/', null=True, blank=True)
    business_address = models.CharField(max_length=255)
    state = models.ForeignKey(State, on_delete=models.SET_NULL, null=True)
    town = models.ForeignKey(Town, on_delete=models.SET_NULL, null=True)
    is_confirmed = models.BooleanField(default=False)
    is_rejected = models.BooleanField(default=False)

    def __str__(self):
        return self.user.email