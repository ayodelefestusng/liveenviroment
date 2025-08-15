from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect, get_object_or_404
from django.http import (
    HttpResponse,
    JsonResponse,
    HttpResponseRedirect,
    HttpResponseServerError
)
from django.urls import reverse
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.contrib import messages

import pyotp
import qrcode
import random
import io
from io import BytesIO
import base64

from django.contrib.auth import get_user_model


from .forms import RegistrationForm,User,PasswordChangeForm,PasswordResetForm,PasswordSetupForm


from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required



def home  (request):
    return render(request, "home.html")
    # return HttpResponse("I am okay")


def check_username(request):

    if request.method == "GET":
        return HttpResponse("Oya")
    elif request.method == "POST":
        email = request.POST.get('email')
        print("AJADI", email)

        if email and User.objects.filter(email=email).exists():
            return HttpResponse("This username already exists")
        return HttpResponse("")  # Empty response if email is available or not provided


@csrf_exempt
def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(None)  # User sets password later
            user.save()

            token = default_token_generator.make_token(user)
            link = request.build_absolute_uri(reverse("setup_password", args=[user.pk, token]))

            send_mail(
                "Set Your Password",
                f"Click the link to set your password: {link}",
                "admin@example.com",
                [user.email],
            )

            # return render(request, "myapp/registration_success.html", {"email": user.email})
            return render(request, "registration/password_setup_sent.html", {"email": user.email})
    else:
        form = RegistrationForm()
    return render(request, "registration/register.html", {"form": form})

@csrf_exempt
def setup_password(request, user_id, token):
    user = User.objects.get(pk=user_id)
    if default_token_generator.check_token(user, token):
        if request.method == "POST":
            form = PasswordSetupForm(user, request.POST)
            if form.is_valid():
                form.save()
                return redirect("login")
                # return HttpResponseRedirect("login")
        else:
            form = PasswordSetupForm(user)
        return render(request, "registration/setup_password.html", {"form": form})
    else:
        return render(request, "registration/error.html", {"message": "Invalid token"})


@csrf_exempt
def password_reset_request(request):
    if request.method == "POST":
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            user = User.objects.filter(email=email).first()
            if user:
                token = default_token_generator.make_token(user)
                link = request.build_absolute_uri(reverse("setup_password", args=[user.pk, token]))
                
                send_mail(
                    "Reset Your Password",
                    f"Click the link to reset your password: {link}",
                    "admin@example.com",
                    [email],
                )
            return render(request, "registration/password_reset_sent.html", {"email": email})
    else:
        form = PasswordResetForm()
    return render(request, "registration/password_reset.html", {"form": form})


@csrf_exempt
def change_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.POST)
        if form.is_valid():
            user = authenticate(email=request.user.email, password=form.cleaned_data["old_password"])
            if user:
                user.set_password(form.cleaned_data["new_password"])
                user.save()
                logout(request)
                return redirect("login")
            else:
                return render(request, "myapp/change_password.html", {"form": form, "error": "Incorrect password"})
    else:
        form = PasswordChangeForm()
    return render(request, "registration/change_password.html", {"form": form})

@csrf_exempt
def user_login(request):
 
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(request, email=email, password=password)

        if user is not None:
            print("ijaya",user.mfa_secret)
            if not user.mfa_enabled:
                # Generate MFA secret and QR code
                user.mfa_secret = pyotp.random_base32()
                user.save()

                otp_uri = pyotp.totp.TOTP(user.mfa_secret).provisioning_uri(
                    name=user.email,
                    issuer_name="RELUCENT"
                )

                qr = qrcode.make(otp_uri)
                buffer = io.BytesIO()
                qr.save(buffer, format="PNG")
                buffer.seek(0)
                qr_code = base64.b64encode(buffer.getvalue()).decode("utf-8")
                qr_code_data_uri = f"data:image/png;base64,{qr_code}"

                # Show QR code for first-time setup
                return render(request, 'registration/profile.html', {"qrcode": qr_code_data_uri,"email": email})

            # Redirect to OTP verification without logging in yet
            # return redirect(reverse("verify_mfa", kwargs={"email": email}))
            # return redirect(reverse("verify_mfa"))
            
            return render(request, 'registration/otp_verify.html', {"email": email})
        messages.error(request, "Invalid email or password. Please try again.",)
            
    
    return render(request, "registration/login.html")


@login_required
def profile_view(request):
        email = request.POST.get("email")
        user = User.objects.get(email=email)
        print("Ajayi", user)
        if not user.mfa_secret:
            user.mfa_secret = pyotp.random_base32()
            user.save()

        otp_uri = pyotp.totp.TOTP(user.mfa_secret).provisioning_uri(
            name=user.email,
            issuer_name="AGOBA DIGNITY"
        )

        qr = qrcode.make(otp_uri)
        buffer = io.BytesIO()
        qr.save(buffer, format="PNG")
        
       
        buffer.seek(0)  
        qr_code = base64.b64encode(buffer.getvalue()).decode("utf-8")

        qr_code_data_uri = f"data:image/png;base64,{qr_code}"
        return render(request, 'registration/profile.html', {"qrcode": qr_code_data_uri})


def verify_2fa_otp(user, otp):
    totp = pyotp.TOTP(user.mfa_secret)
    if totp.verify(otp):
        user.mfa_enabled = True
        user.save()
        return True
    return False


def verify_mfa(request):
    # email = request.POST.get('email')
    # print ("aleko",request)

    # try:
    #     user = User.objects.get(email=email)
    # except User.DoesNotExist:
    #     messages.error(request, 'User not founds.')
    #     return redirect('login')

    if request.method == 'POST':
        otp = request.POST.get('otp_code')
        print ("otp",otp)
        email = request.POST.get('email')
        print ("email",email)

        user = User.objects.get(email=email)
       
        

        if verify_2fa_otp(user, otp):
            login(request, user)  # ✅ Only log in after successful OTP
            messages.success(request, 'Login successful with 2FA!')
            return redirect('home')
        else:
            messages.error(request, 'Invalid OTP code. Please try again.')
            return render(request, 'registration/otp_verify.html', {'email': email})

    return render(request, 'registration/otp_verify.html', {'email': email})




def reset_qr(request):
    if request.method == 'POST':
   
        email = request.POST.get('email')
        user=User.objects.get(email=email)
        user.mfa_enabled=False
        user.save()
    
        otp_uri = pyotp.totp.TOTP(user.mfa_secret).provisioning_uri(
            name=email,
            issuer_name="AGOBA DIGNITY"
        )

        qr = qrcode.make(otp_uri)
        buffer = io.BytesIO()
        qr.save(buffer, format="PNG")
        
       
        buffer.seek(0)  
        qr_code = base64.b64encode(buffer.getvalue()).decode("utf-8")

        qr_code_data_uri = f"data:image/png;base64,{qr_code}"

        return render(request, 'registration/profile.html', {"qrcode": qr_code_data_uri})

    



@csrf_exempt
@login_required

def user_logout(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("login")  # Redirect to login page after logout
