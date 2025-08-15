
# Register your models here.
from django.contrib import admin
from .models import Currency,Country,Client,BranchDetails,TellerDetails,Customer,Transaction,BranchAccountTill

# Register your models here.

admin.site.register(Currency)
admin.site.register(Country)
admin.site.register(Client)
admin.site.register(BranchDetails)
admin.site.register(TellerDetails)
admin.site.register(Customer)
admin.site.register(Transaction)
admin.site.register(BranchAccountTill)



