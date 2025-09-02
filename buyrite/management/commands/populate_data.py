import csv
from django.core.management.base import BaseCommand
from django.db import transaction
from myapp.models import Category, Brand, VehicleModel, Trim # Replace 'your_app_name'

class Command(BaseCommand):
    help = 'Populates Category, Brand, VehicleModel, and Trim models from a CSV file.'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The path to the CSV file to import.')

    def handle(self, *args, **options):
        file_path = options['csv_file']
        self.stdout.write(self.style.SUCCESS(f'Starting data import from {file_path}...'))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                
                # Clean up the fieldnames to remove any whitespace
                reader.fieldnames = [header.strip() for header in reader.fieldnames]

                # Dictionaries to store unique instances to be created
                categories_to_create = {}
                brands_to_create = {}
                models_to_create = {}
                trims_to_create = []

                # Look up existing objects to prevent duplicates
                existing_categories = {obj.name: obj for obj in Category.objects.all()}
                existing_brands = {obj.name: obj for obj in Brand.objects.all()}
                existing_models = {obj.name: obj for obj in VehicleModel.objects.all()}
                
                for row in reader:
                    category_name = row.get('Category', '').strip()
                    brand_name = row.get('Brand', '').strip()
                    model_name = row.get('Model', '').strip()
                    trim_name = row.get('Trim', 'N/A').strip()

                    # Collect unique categories
                    if category_name and category_name not in existing_categories and category_name not in categories_to_create:
                        categories_to_create[category_name] = Category(name=category_name)
                    
                    # Collect unique brands
                    if brand_name and brand_name not in existing_brands and brand_name not in brands_to_create:
                        brands_to_create[brand_name] = Brand(name=brand_name)
                    
                    # Collect unique models
                    if model_name and model_name not in existing_models and model_name not in models_to_create:
                        models_to_create[model_name] = {'name': model_name, 'brand_name': brand_name}

                    # Collect all trims
                    if trim_name:
                        trims_to_create.append({'trim_name': trim_name, 'model_name': model_name})

                # --- Bulk Creation in an Atomic Transaction ---
                with transaction.atomic():
                    # Bulk create categories
                    new_categories = Category.objects.bulk_create(list(categories_to_create.values()))
                    for obj in new_categories:
                        existing_categories[obj.name] = obj

                    # Bulk create brands
                    new_brands = Brand.objects.bulk_create(list(brands_to_create.values()))
                    for obj in new_brands:
                        existing_brands[obj.name] = obj

                    # Bulk create vehicle models
                    model_objects = []
                    for model_data in models_to_create.values():
                        brand_instance = existing_brands.get(model_data['brand_name'])
                        if brand_instance:
                            model_objects.append(VehicleModel(name=model_data['name'], brand=brand_instance))
                    
                    new_models = VehicleModel.objects.bulk_create(model_objects)
                    for obj in new_models:
                        existing_models[obj.name] = obj
                    
                    # Bulk create trims
                    trim_objects = []
                    for trim_data in trims_to_create:
                        model_instance = existing_models.get(trim_data['model_name'])
                        if model_instance:
                            trim_objects.append(Trim(name=trim_data['trim_name'], vehicle_model=model_instance))
                    
                    Trim.objects.bulk_create(trim_objects, ignore_conflicts=True) # ignore_conflicts for duplicate entries

            self.stdout.write(self.style.SUCCESS('Data population complete!'))
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'Error: The file {file_path} was not found.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred: {e}"))