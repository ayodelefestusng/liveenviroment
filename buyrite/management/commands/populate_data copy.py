# my_app/management/commands/populate_data.py
import csv
from django.core.management.base import BaseCommand
from django.db import transaction
from myapp.models import State, Town # Replace with your app name

class Command(BaseCommand):
    help = 'Populates the State and Town models from a CSV file.'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The path to the CSV file to import.')

    def handle(self, *args, **options):
        file_path = options['csv_file']
        self.stdout.write(self.style.SUCCESS(f'Starting data import from {file_path}...'))

        try:
            with open(file_path, 'r', encoding='cp1252') as csv_file:
                reader = csv.DictReader(csv_file)
                # This line remaps the fieldnames, stripping whitespace from them.
                reader.fieldnames = [header.strip() for header in reader.fieldnames]

                states_to_create = {}
                towns_to_create = []
                existing_states = {state.name: state for state in State.objects.all()}

                for row in reader:
                    state_name = row['State'].strip()
                    town_name = row['Town'].strip()

                    if state_name not in existing_states and state_name not in states_to_create:
                        states_to_create[state_name] = State(name=state_name)

                    towns_to_create.append({'state_name': state_name, 'town_name': town_name})

                with transaction.atomic():
                    new_states = State.objects.bulk_create(list(states_to_create.values()))

                    for state in new_states:
                        existing_states[state.name] = state

                    town_objects = []
                    for town_data in towns_to_create:
                        state_instance = existing_states[town_data['state_name']]
                        town_objects.append(Town(name=town_data['town_name'], state=state_instance))

                    Town.objects.bulk_create(town_objects)

            self.stdout.write(self.style.SUCCESS('Data population complete!'))
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'Error: The file {file_path} was not found.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'An error occurred: {e}'))