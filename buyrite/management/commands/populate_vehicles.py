# import os
# import random
# from django.core.files.base import ContentFile
# from django.core.management.base import BaseCommand
# from django.core.files import File
# from django.contrib.auth.hashers import make_password
# from myapp.models import (
#     Brand, Category, ManufactureYear, User, VehicleModel, Trim, Vehicle,
#     FuelOption, Color, EngineType, DriveTerrain, State, Town, Vas
# )
# from myapp.models import Condition
# from django.utils.text import slugify


# class Command(BaseCommand):
#     help = 'Auto-populate vehicle eCommerce database with images and randomized attributes'

#     def handle(self, *args, **kwargs):
#         image_paths = [
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Volvo-XC70_CN-Version-2026-thb.jpg',
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Volkswagen v1.jpg',
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Ford-Bronco_2-door-2021-thb.jpg',
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Ford-Crown_Victoria  v3.jpg',
#             r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Ford-Crown_Victoria  v3.jpg'

#         ]


#         transmission_choices = ['automatic', 'manual']
#         # conditions = ['New', 'Used', 'Refurbished']
  
#         conditions = list(Condition.objects.all())


#         brands = ['Toyota', 'Nissan', 'Honda', 'Mercedez Benz', 'Bajaj', 'Cateripillar', 'Volvo']
#         brand_objs = [Brand.objects.get_or_create(name=b)[0] for b in brands]

#         categories = ['Car', 'Bus', 'Lorry', 'Machinery', 'Motorcycle', 'Truck', 'Spare Part']
#         category_objs = [Category.objects.get_or_create(name=c)[0] for c in categories]

#         years = list(range(2005, 2025))
#         year_objs = [ManufactureYear.objects.get_or_create(year=y)[0] for y in years]

#         sellers = ['yemi', 'kunle', 'timothy']
#         seller_objs = []
#         for name in sellers:
#             user, _ = User.objects.get_or_create(
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

#         # for brand in brand_objs:
#         #     for i in range(1, 4):
#         #         model_name = f"{brand.name} Model {i}"
#         #         model = VehicleModel.objects.get_or_create(name=model_name, brand=brand)[0]
#         #         for j in ['Base', 'Sport', 'Luxury']:
#         #             Trim.objects.get_or_create(name=j, vehicle_model=model)

#         # trims = list(Trim.objects.all())
#         # fuels = list(FuelOption.objects.all())
#         # colors = list(Color.objects.all())
#         # engines = list(EngineType.objects.all())
#         # terrains = list(DriveTerrain.objects.all())
#         # states = list(State.objects.all())
#         # towns = list(Town.objects.all())
#         # vas_options = list(Vas.objects.all())
#         for brand in brand_objs:
#             for i in range(1, 4):
#                 model_name = f"{brand.name} Model {i}"
#                 model = VehicleModel.objects.get_or_create(name=model_name, brand=brand)[0]
#                 for j in ['Base', 'Sport', 'Luxury']:
#                     Trim.objects.get_or_create(name=j, vehicle_model=model)

#         # Move the population of trims here, AFTER they have been created.
#         trims = list(Trim.objects.all())
#         fuels = list(FuelOption.objects.all())
#         colors = list(Color.objects.all())
#         engines = list(EngineType.objects.all())
#         terrains = list(DriveTerrain.objects.all())
#         states = list(State.objects.all())
#         towns = list(Town.objects.all())
#         vas_options = list(Vas.objects.all())

#         if not all([trims, fuels, colors, engines, terrains, states, towns, vas_options]):
#             self.stdout.write(self.style.WARNING('⚠️ One or more required tables are empty. Please seed them before running this command.'))
#             return

#         self.stdout.write(self.style.WARNING('🗑️ Deleting all existing vehicle records...'))
#         Vehicle.objects.all().delete()
#         self.stdout.write(self.style.SUCCESS('✅ All existing vehicle records deleted.'))

#         for _ in range(30):
#             trim = random.choice(trims)
#             vehicle_model = trim.vehicle_model
#             brand = vehicle_model.brand
#             category = random.choice(category_objs)
#             year = random.choice(year_objs)
#             seller = random.choice(seller_objs)
#             # condition = random.choice(conditions)
#             if conditions:
#                 condition = random.choice(conditions)
#             else:
#                 self.stdout.write(self.style.WARNING('⚠️ No conditions found. Skipping vehicle creation.'))
#                 continue
#             fuel = random.choice(fuels)
#             color = random.choice(colors)
#             engine = random.choice(engines)
#             terrain = random.choice(terrains)
#             state = random.choice(states)

#             towns_in_state = [t for t in towns if t.state == state]
#             town = random.choice(towns_in_state) if towns_in_state else random.choice(towns)

#             vas_selected = random.sample(vas_options, k=random.randint(1, 3))
#             selected_images = random.choices(image_paths, k=10)

#             vehicle = Vehicle(
#                 seller=seller,
#                 category=category,
#                 brand=brand,
#                 vehicle_model=vehicle_model,
#                 trim=trim,
#                 manufacture_year=year,
#                 condition=condition,
#                 fuel_option=fuel,
#                 color=color,
#                 engine_type=engine,
#                 drive_terrain=terrain,
#                 state=state,
#                 town=town,
#                 exchange_option='no',
#                 registered='no',
#                 negotiable='yes',
#                 no_of_tyres=random.randint(4, 12),
#                 seat=random.randint(2, 60),
#                 mileage=random.randint(10000, 150000),
#                 price=random.randint(2000000, 15000000),
#                 contact_phone='08012345678',
#                 description=f"{brand.name} {vehicle_model.name} {trim.name} - Reliable and affordable.",
#                 is_available=True,
#                 transmission=random.choice(transmission_choices),
#                 number_cylinder=random.choice([None] + list(range(1, 21))),
#                 horsepower=random.choice([None] + list(range(50, 201)))
#             )

#             base_slug = slugify(f"{vehicle.brand.name}-{vehicle.vehicle_model.name}")
#             slug = base_slug
#             counter = 1
#             while Vehicle.objects.filter(slug=slug).exists():
#                 slug = f"{base_slug}-{counter}"
#                 counter += 1
#             vehicle.slug = slug

#             base_index = slugify(f"{vehicle.brand.name}-{vehicle.trim.name}-{vehicle.category.name}-"
#                                  f"{vehicle.manufacture_year.year}-{vehicle.vehicle_model.name}-"
#                                  f"{vehicle.condition.name}-{vehicle.fuel_option.name}-"
#                                  f"{vehicle.engine_type.name}-{vehicle.drive_terrain.name}-"
#                                  f"{vehicle.state.name}-{vehicle.town.name}")
#             index = base_index
#             counter = 1
#             while Vehicle.objects.filter(index=index).exists():
#                 index = f"{base_index}-{counter}"
#                 counter += 1
#             vehicle.index = index
            

#             valid_image_paths = [p for p in selected_images if os.path.exists(p)]
#             valid_image_paths = [p for p in selected_images if os.path.exists(p)]
#             if len(valid_image_paths) < 5:
#                 self.stdout.write(self.style.WARNING("⚠️ Not enough valid images. Skipping vehicle."))
#                 continue
#             # if len(valid_image_paths) < 5:
#             #     self.stdout.write(self.style.WARNING("⚠️ Not enough valid images. Skipping vehicle."))
#             #     continue

#             # ✅ This part must be outside the `if` block
#             # for idx, image_path in enumerate(valid_image_paths[:10]):
#             #     with open(image_path, 'rb') as img_file:
#             #         image_data = img_file.read()
#             #         image_file = ContentFile(image_data, name=os.path.basename(image_path))
#             #         setattr(vehicle, f'image{idx+1}', image_file)

#             # required_images = ['image', 'image2', 'image3', 'image4', 'image5']
#             # missing = [field for field in required_images if not getattr(vehicle, field)]

#             # if missing:
#             #     self.stdout.write(self.style.WARNING(f"⚠️ Skipping vehicle due to missing images: {missing}"))
#             #     continue


#             # ... existing code ...

#             for idx, image_path in enumerate(valid_image_paths[:10]):
#                 with open(image_path, 'rb') as img_file:
#                     image_data = img_file.read()
#                     image_file = ContentFile(image_data, name=os.path.basename(image_path))

#                     if idx == 0:
#                         setattr(vehicle, 'image', image_file)  # Set the 'image' field for the first image
#                     else:
#                         setattr(vehicle, f'image{idx+1}', image_file) # Set image2, image3, etc.

#             required_images = ['image', 'image2', 'image3', 'image4', 'image5']
#             missing = [field for field in required_images if not getattr(vehicle, field)]

#             if missing:
#                 self.stdout.write(self.style.WARNING(f"⚠️ Skipping vehicle due to missing images: {missing}"))
#                 continue

#             vehicle.save()
#             vehicle.vas.set(vas_selected)
#             # vehicle.vas.set(vas_selected)

#         self.stdout.write(self.style.SUCCESS('✅ Vehicle database fully populated with randomized attributes and images.'))


import os
import random
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from django.contrib.auth.hashers import make_password
from myapp.models import (
    Brand, Category, ManufactureYear, VehicleModel, Trim, Vehicle,
    FuelOption, Color, EngineType, DriveTerrain, State, Town, Vas
)
from users.models import User
from myapp.models import Condition
from django.utils.text import slugify
from django.conf import settings


class Command(BaseCommand):
    help = 'Auto-populate vehicle eCommerce database with images and randomized attributes'

    def handle(self, *args, **kwargs):
        # Image paths
        image_paths = [
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Volvo-XC70_CN-Version-2026-thb.jpg',
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Volkswagen v1.jpg',
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Ford-Bronco_2-door-2021-thb.jpg',
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Ford-Crown_Victoria  v3.jpg',
            r'C:\Users\Pro\Desktop\PROJECT\Buy_Rite\myproject\static\images\New folder\car\Ford-Crown_Victoria  v3.jpg'
        ]

        valid_image_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_image_paths:
            self.stdout.write(self.style.ERROR("🚨 No valid image paths found. Please check the paths in your script."))
            return

        transmission_choices = ['automatic', 'manual']

        try:
            conditions = list(Condition.objects.all())
            if not conditions:
                self.stdout.write(self.style.WARNING('⚠️ No conditions found. Please seed the Condition table.'))
                return
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"🚫 Error fetching Conditions: {e}"))
            return

        brands = ['Toyota', 'Nissan', 'Honda', 'Mercedez Benz', 'Bajaj', 'Cateripillar', 'Volvo']
        brand_objs = [Brand.objects.get_or_create(name=b)[0] for b in brands]

        categories = ['Car', 'Bus', 'Lorry', 'Machinery', 'Motorcycle', 'Truck', 'Spare Part']
        category_objs = [Category.objects.get_or_create(name=c)[0] for c in categories]

        years = list(range(2005, 2025))
        year_objs = [ManufactureYear.objects.get_or_create(year=y)[0] for y in years]

        # Use the specific email addresses to create or get User objects
        seller_emails = ['dignity.upwardwave@gmail.com', 'ayodelefestusng@gmail.com',"kunle@gmail.com", "wale@gmail.com"]
        seller_objs = []
        for email in seller_emails:
            user, _ = User.objects.get_or_create(
                email=email,
                defaults={
                    'full_name': email.split('@')[0].replace('.', ' ').title(),
                    'phone': '08012345678',
                    'is_seller': True,
                    'is_buyer': False,
                    'is_active': True,
                    'password': make_password('Ajibandele')
                }
            )
            seller_objs.append(user)

        # Create vehicle models and trims
        for brand in brand_objs:
            for i in range(1, 4):
                model_name = f"{brand.name} Model {i}"
                model = VehicleModel.objects.get_or_create(name=model_name, brand=brand)[0]
                for j in ['Base', 'Sport', 'Luxury']:
                    Trim.objects.get_or_create(name=j, vehicle_model=model)

        try:
            trims = list(Trim.objects.all())
            fuels = list(FuelOption.objects.all())
            colors = list(Color.objects.all())
            engines = list(EngineType.objects.all())
            terrains = list(DriveTerrain.objects.all())
            states = list(State.objects.all())
            towns = list(Town.objects.all())
            vas_options = list(Vas.objects.all())
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"🚫 Error fetching model objects: {e}"))
            return
            
        if not all([trims, fuels, colors, engines, terrains, states, towns, vas_options, seller_objs]):
            self.stdout.write(self.style.WARNING('⚠️ One or more required tables are empty. Please seed them before running this command.'))
            return

        self.stdout.write(self.style.WARNING('🗑️ Deleting all existing vehicle records...'))
        Vehicle.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('✅ All existing vehicle records deleted.'))

        for _ in range(30):
            try:
                trim = random.choice(trims)
                vehicle_model = trim.vehicle_model
                brand = vehicle_model.brand
                category = random.choice(category_objs)
                year = random.choice(year_objs)
                
                # Get a seller object directly from the list of user instances
                seller = random.choice(seller_objs)
                
                condition = random.choice(conditions)
                fuel = random.choice(fuels)
                color = random.choice(colors)
                engine = random.choice(engines)
                terrain = random.choice(terrains)
                state = random.choice(states)
                
                towns_in_state = [t for t in towns if t.state == state]
                town = random.choice(towns_in_state) if towns_in_state else random.choice(towns)

                vas_selected = random.sample(vas_options, k=random.randint(1, 3))
                
                if len(valid_image_paths) < 5:
                    self.stdout.write(self.style.WARNING("⚠️ Not enough valid images. Skipping vehicle."))
                    continue
                
                selected_images = random.choices(valid_image_paths, k=5)

                vehicle = Vehicle(
                    seller=seller, # Correctly assigns a User object
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
                
                for idx, image_path in enumerate(selected_images):
                    with open(image_path, 'rb') as img_file:
                        image_data = img_file.read()
                        image_file = ContentFile(image_data, name=os.path.basename(image_path))
                        
                        if idx == 0:
                            vehicle.image = image_file
                        else:
                            setattr(vehicle, f'image{idx+1}', image_file)
                
                required_images = ['image', 'image2', 'image3', 'image4', 'image5']
                missing_images = [field for field in required_images if not getattr(vehicle, field, False)]

                if missing_images:
                    self.stdout.write(self.style.WARNING(f"⚠️ Skipping vehicle due to missing images: {missing_images}"))
                    continue

                vehicle.save()
                vehicle.vas.set(vas_selected)
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"🚫 An error occurred while creating a vehicle: {e}"))
                continue

        self.stdout.write(self.style.SUCCESS('✅ Vehicle database fully populated with randomized attributes and images.'))