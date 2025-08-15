from django.urls import path

from . import views

from django.urls import path

from . import views

# urlpatterns = [
#     # ex: /polls/
#     path("", views.index, name="index"),
#     # ex: /polls/5/
#     path("<int:question_id>/", views.detail, name="detail"),
#     # ex: /polls/5/results/
#     path("<int:question_id>/results/", views.results, name="results"),
#     # ex: /polls/5/vote/
#     path("<int:question_id>/vote/", views.vote, name="vote"),
# ]


from django.urls import path
from .views import *


urlpatterns = [
    path("",home, name="home"),



#Transaction 

    path('currency/', views.add_currency, name='currency'),
    path('country/', views.add_country, name='country'),

#Teller 
# path("account_lookup/", views.sender_with_account_lookup, name="sender_with_account_lookup"),
       

 
    path("teller/account_lookup/", views.sender_with_account_lookup, name="sender_with_account_lookup"),
     path("teller/transaction-without-account/", views.transaction_without_account, name="transaction_without_account"),
        path('contact/success/', lambda request: render(request, 'success.html'), name='contact_success'),
]

hmtx_views = [

    path("check_account/", views.check_account, name='check_account'),
    path("balance_check/", views.balance_check, name='balance_check'),
        path("teller/transaction-with-account/", views.transaction_with_account, name="transaction_with_account"),

]

urlpatterns += hmtx_views
