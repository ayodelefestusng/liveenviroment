# buyrite/context_processors.py
from .forms import VINCheckForm

def vin_form_processor(request):
    return {'vin_form': VINCheckForm()}