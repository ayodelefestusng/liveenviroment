
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


from .forms import RegistrationForm,CustomUser,PasswordChangeForm,PasswordResetForm,PasswordSetupForm


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

            return render(request, "myapp/registration_success.html", {"email": user.email})
    else:
        form = RegistrationForm()
    return render(request, "myapp/register.html", {"form": form})

def setup_password(request, user_id, token):
    user = CustomUser.objects.get(pk=user_id)
    if default_token_generator.check_token(user, token):
        if request.method == "POST":
            form = PasswordSetupForm(user, request.POST)
            if form.is_valid():
                form.save()
                return redirect("login")
        else:
            form = PasswordSetupForm(user)
        return render(request, "myapp/setup_password.html", {"form": form})
    else:
        return render(request, "myapp/error.html", {"message": "Invalid token"})

def password_reset_request(request):
    if request.method == "POST":
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            user = CustomUser.objects.filter(email=email).first()
            if user:
                token = default_token_generator.make_token(user)
                link = request.build_absolute_uri(reverse("setup_password", args=[user.pk, token]))
                
                send_mail(
                    "Reset Your Password",
                    f"Click the link to reset your password: {link}",
                    "admin@example.com",
                    [email],
                )
            return render(request, "myapp/templates/password_reset_sent.html", {"email": email})
    else:
        form = PasswordResetForm()
    return render(request, "myapp/password_reset.html", {"form": form})

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
    return render(request, "myapp/change_password.html", {"form": form})



def user_login(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(request, email=email, password=password)

        if user:
            login(request, user)
            return redirect("home")  # Redirect to dashboard after login
        else:
            messages.error(request, "Invalid email or password. Please try again.")

    return render(request, "myapp/login.html")


@login_required
def user_logout(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("login")  # Redirect to login page after logout

