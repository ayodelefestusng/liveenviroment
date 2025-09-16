from django.core.mail import send_mail


import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
django.setup()

send_mail(
    subject="Test Email Heeejejej",
    message="Hello Ayodele, this is a test email from your Django app!",
    from_email="demos@kupiansolutions.com",
    recipient_list=["upwardwave.dignity@gmail.com"],
    fail_silently=False,
)