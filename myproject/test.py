import os
import sys

import django
from django.core.mail import send_mail

# Add project directory to Python path
sys.path.append("C:/Users/Pro/Desktop/PROJECT/Live/myproject")

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
django.setup()

print("Starting email send...")

# Send the email
send_mail(
    subject="Test Email",
    message="Hello Ayodele, this is a test email from your Django app!",
    from_email="demos@kupiansolutions.com",
    recipient_list=["ayodelefestusng@mail.com"],
    fail_silently=False,
)

print("Email sent successfully!")