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


    #Account Management 
     path("check-username/", views.check_username, name='check-username'),

     path("register/", register, name="register"),
    path("setup-password/<int:user_id>/<str:token>/", setup_password, name="setup_password"),
    path("password-reset/", password_reset_request, name="password_reset"),
    path("change-password/", change_password, name="change_password"),
    path("login/", user_login, name="login"),
    path("logout/", user_logout, name="logout"),
#   path('verify_mfa/<str:email>/', verify_mfa, name='verify_mfa'),
  path('verify_mfa/', verify_mfa, name='verify_mfa'),

  path('reset_qr/', reset_qr, name='reset_qr'),
    # path('disable-2fa/', disable_2fa, name='disable_2fa'),



]

hmtx_views = [
    path("check_username/", views.check_username, name='check_username'),
 

]

urlpatterns += hmtx_views
