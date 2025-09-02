import os
import random
from django.core.management.base import BaseCommand
from django.core.files import File
from django.contrib.auth.hashers import make_password
from myapp.models import Brand, Category, ManufactureYear, Trim, VehicleModel, Vehicle, User

# class Command(BaseCommand):
#     help = 'Auto-populate vehicle eCommerce database with images'

#     def handle(self, *args, **kwargs):
#         # Image paths to randomly assign to vehicles
#         image_paths = [
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\media\category_images\bus2.png',
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\media\category_images\car.png',
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\media\category_images\heavy-equipment.png',
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\media\category_images\motorcycle.png'
#         ]

#         # Brands
#         brands = ['Toyota', 'Nissan', 'Honda', 'Mercedez Benz', 'Bajaj', 'Cateripillar', 'Volvo']
#         brand_objs = [Brand.objects.get_or_create(name=b)[0] for b in brands]

#         # Categories
#         categories = ['Car', 'Bus', 'Lorry', 'Machinery', 'Motorcycle', 'Truck', 'Spare Part']
#         category_objs = [Category.objects.get_or_create(name=c)[0] for c in categories]

#         # Manufacture Years
#         years = list(range(2005, 2025))
#         year_objs = [ManufactureYear.objects.get_or_create(year=y)[0] for y in years]

#         # Sellers
#         sellers = ['yemi', 'kunle', 'timothy']
#         seller_objs = []
#         for name in sellers:
#             user, created = User.objects.get_or_create(
#                 email=f"{name}@example.com",
#                 defaults={
#                     'full_name': name.title(),
#                     'phone': '08012345678',
#                     'is_seller': True,
#                     'is_buyer': False,
#                     'is_active': True,
#                     'password': make_password('Ajibandele')
#                 }
#             )
#             seller_objs.append(user)

#         # Vehicle Models and Trims
#         for brand in brand_objs:
#             for i in range(1, 4):
#                 model_name = f"{brand.name} Model {i}"
#                 model = VehicleModel.objects.get_or_create(name=model_name, brand=brand)[0]
#                 for j in ['Base', 'Sport', 'Luxury']:
#                     Trim.objects.get_or_create(name=j, vehicle_model=model)

#         # Vehicles
#         trims = Trim.objects.all()
#         for _ in range(30):
#             trim = random.choice(trims)
#             vehicle_model = trim.vehicle_model
#             brand = vehicle_model.brand
#             category = random.choice(category_objs)
#             year = random.choice(year_objs)
#             seller = random.choice(seller_objs)
#             image_path = random.choice(image_paths)

#             vehicle = Vehicle.objects.create(
#                 seller=seller,
#                 category=category,
#                 brand=brand,
#                 vehicle_model=vehicle_model,
#                 trim=trim,
#                 manufacture_year=year,
#                 mileage=random.randint(10000, 150000),
#                 price=random.randint(2000000, 15000000),
#                 description=f"{brand.name} {vehicle_model.name} {trim.name} - Reliable and affordable.",
#                 is_available=True
#             )

#             # Attach image if file exists
#             if os.path.exists(image_path):
#                 with open(image_path, 'rb') as img_file:
#                     vehicle.image.save(os.path.basename(image_path), File(img_file), save=True)

#         self.stdout.write(self.style.SUCCESS('✅ Vehicle database populated successfully with images.'))


from django.core.management.base import BaseCommand
from django.core.files import File
from django.contrib.auth.hashers import make_password
from myapp.models import (
    Brand, Category, ManufactureYear, Trim, VehicleModel, Vehicle, User,
    Condition, FuelOption, Color, EngineType, DriveTerrain, State, Town, Vas
)
import os
import random

from django.core.management.base import BaseCommand
from django.core.files import File
from django.contrib.auth.hashers import make_password
from django.utils.text import slugify
from myapp.models import (
    Vehicle, Brand, Category, ManufactureYear, VehicleModel, Trim,
    Condition, FuelOption, Color, EngineType, DriveTerrain, State, Town, Vas,User
)

import os, random

class Command(BaseCommand):
    help = 'Auto-populate vehicle eCommerce database with images and randomized attributes'

    def handle(self, *args, **kwargs):
        image_paths = [
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\media\category_images\bus2.png',
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\media\category_images\car.png',
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\media\category_images\heavy-equipment.png',
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\media\category_images\motorcycle.png'
        ]

        transmission_choices = ['automatic', 'manual']

        brands = ['Toyota', 'Nissan', 'Honda', 'Mercedez Benz', 'Bajaj', 'Cateripillar', 'Volvo']
        brand_objs = [Brand.objects.get_or_create(name=b)[0] for b in brands]

        categories = ['Car', 'Bus', 'Lorry', 'Machinery', 'Motorcycle', 'Truck', 'Spare Part']
        category_objs = [Category.objects.get_or_create(name=c)[0] for c in categories]

        years = list(range(2005, 2025))
        year_objs = [ManufactureYear.objects.get_or_create(year=y)[0] for y in years]

        sellers = ['yemi', 'kunle', 'timothy']
        seller_objs = []
        for name in sellers:
            user, _ = User.objects.get_or_create(
                email=f"{name}@example.com",
                defaults={
                    'full_name': name.title(),
                    'phone': '08012345678',
                    'is_seller': True,
                    'is_buyer': False,
                    'is_active': True,
                    'password': make_password('Ajibandele')
                }
            )
            seller_objs.append(user)

        for brand in brand_objs:
            for i in range(1, 4):
                model_name = f"{brand.name} Model {i}"
                model = VehicleModel.objects.get_or_create(name=model_name, brand=brand)[0]
                for j in ['Base', 'Sport', 'Luxury']:
                    Trim.objects.get_or_create(name=j, vehicle_model=model)

        trims = Trim.objects.all()
        # conditions = list(Condition.objects.all())
        conditions = ['New', 'Used', 'Refurbished']
        fuels = list(FuelOption.objects.all())
        colors = list(Color.objects.all())
        engines = list(EngineType.objects.all())
        terrains = list(DriveTerrain.objects.all())
        states = list(State.objects.all())
        towns = list(Town.objects.all())
        vas_options = list(Vas.objects.all())

        for _ in range(30):
            trim = random.choice(trims)
            vehicle_model = trim.vehicle_model
            brand = vehicle_model.brand
            category = random.choice(category_objs)
            year = random.choice(year_objs)
            seller = random.choice(seller_objs)
            # condition = random.choice(conditions)
            if conditions:
                condition = random.choice(conditions)
            else:
                condition = 'Unknown'  # or any default value   
            fuel = random.choice(fuels)
            color = random.choice(colors)
            engine = random.choice(engines)
            terrain = random.choice(terrains)
            state = random.choice(states)
            town = random.choice([t for t in towns if t.state == state] or towns)
            vas_selected = random.sample(vas_options, k=random.randint(1, 3))
            selected_images = random.choices(image_paths, k=10)

            vehicle = Vehicle(
                seller=seller,
                category=category,
                brand=brand,
                vehicle_model=vehicle_model,
                trim=trim,
                manufacture_year=year,
                condition=condition,
                fuel_option=fuel,
                color=color,
                engine_type=engine,
                drive_terrain=terrain,
                state=state,
                town=town,
                exchange_option='no',
                registered='no',
                negotiable='yes',
                no_of_tyres=random.randint(4, 12),
                seat=random.randint(2, 60),
                mileage=random.randint(10000, 150000),
                price=random.randint(2000000, 15000000),
                contact_phone='08012345678',
                description=f"{brand.name} {vehicle_model.name} {trim.name} - Reliable and affordable.",
                is_available=True,
                transmission=random.choice(transmission_choices),
                number_cylinder=random.choice([None] + list(range(1, 21))),
                horsepower=random.choice([None] + list(range(50, 201)))
            )

            # Attach images
            for idx, image_path in enumerate(selected_images):
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        setattr(vehicle, f'image{idx+1}', File(img_file))

            vehicle.save()
            vehicle.vas.set(vas_selected)

        self.stdout.write(self.style.SUCCESS('✅ Vehicle database fully populated with randomized attributes and images.'))