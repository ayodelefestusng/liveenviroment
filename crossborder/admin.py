
# Register your models here.
from django.contrib import admin

from .models import (BranchAccountTill, BranchDetails, Client, Country,
                     Currency, Customer, TellerDetails, Transaction)

# Register your models here.

admin.site.register(Currency)
admin.site.register(Country)
admin.site.register(Client)
admin.site.register(BranchDetails)
admin.site.register(TellerDetails)
admin.site.register(Customer)
admin.site.register(Transaction)
admin.site.register(BranchAccountTill)



