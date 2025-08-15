# myproject/myapp/forms.py

from django import forms
from django.contrib.auth.forms import PasswordResetForm, SetPasswordForm, AuthenticationForm
from django.urls import reverse_lazy

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, HTML, Div, Fieldset, Submit

from .models import (
    User,
    # Currency,
    # Country,
    # Client,
    # BranchDetails,
    # TellerDetails,
    # Customer,
    # Transaction,
    # BranchAccountTill,
    # AccountTransactionT,
    # Crispy,
    # validate_nigerian_prefix,
)



class RegistrationForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["email", "full_name" ]
        widgets = {
            'email': forms.EmailInput(attrs={
                'hx-post': reverse_lazy('check_username'),
                'hx-trigger': 'keyup',
                'hx-target': '#username-err'
            }),
        }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_action = reverse_lazy('register')
        self.helper.form_method = ('POST')
        self.helper.add_input(Submit('submit', 'Register'))
        self.helper.layout = Layout(
        
            Field('email'),
            # This is the custom div with id "ayo"
            HTML('<div class="text-danger mt-2" id="username-err">This is the extra div content!</div>'),
             HTML('<div class="custom-divider">-- Divider Between Names --</div>'),
             HTML('<p>More content inside the "ayo" div.</p>'),
            Field('full_name'),
        )

    
    def clean_email(self):
        email = self.cleaned_data.get("email")
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email





class RegistrationForm6666(forms.ModelForm):
    class Meta:
        model = User
        fields = ["email", "full_name" ]
        widgets = {
            'email': forms.EmailInput(attrs={
                'hx-post': reverse_lazy('check_username'),
                'hx-trigger': 'keyup',
                'hx-target': '#username-err'
            }),
        }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_action = reverse_lazy('register')
        self.helper.form_method = ('POST')
        self.helper.add_input(Submit('submit', 'Register'))
        self.helper.layout = Layout(
        
            Field('email'),
            # This is the custom div with id "ayo"
            HTML('<div class="text-danger mt-2" id="username-err">This is the extra div content!</div>'),
             HTML('<div class="custom-divider">-- Divider Between Names --</div>'),
             HTML('<p>More content inside the "ayo" div.</p>'),
            Field('full_name'),
        )

    
    def clean_email(self):
        email = self.cleaned_data.get("email")
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email


class PasswordSetupForm(SetPasswordForm):
    pass

        
class PasswordChangeForm(forms.Form):
    old_password = forms.CharField(widget=forms.PasswordInput)
    new_password = forms.CharField(widget=forms.PasswordInput)
 


# class SenderAccountLookupForm(forms.ModelForm):
#     class Meta:
#         model = Customer
#         fields = ["account_number"]
#         widgets = {
#             "account_number": forms.TextInput(attrs={
#                 "hx-post": reverse_lazy("check_account"),
#                 'hx-trigger': 'keyup',
#                 'hx-target': '#account-err',
#                 # "autocomplete": "off",
#             }),
#         }

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.helper = FormHelper()
#         self.helper.form_action = reverse_lazy('sender_with_account_lookup')
#         self.helper.form_method = ('POST')
#         # self.helper.add_input(Submit('submit', 'Look UP'))
#         self.helper.layout = Layout(
        
#             Field('account_number'),
#             # This is the custom div with id "ayo"
#             HTML('<div class="text-dark  mt-2" id="account-err"></div>'),
#             #  HTML('<div class="custom-divider">-- Divider Between Names --</div>'),
#             #  HTML('<p>More content inside the "ayo" div.</p>'),
#         )



#     def clean_account_number(self):
#         return self.cleaned_data["account_number"].strip()





# class CountryForm(forms.ModelForm):
#     class Meta:
#         model = Country
#         fields = ['name', 'currency']
#         widgets = {
#             'name': forms.TextInput(attrs={'class': 'form-control'}),
#             'currency': forms.Select(attrs={'class': 'form-control'}),
#         }


# class TransactionForm(forms.ModelForm):
#     class Meta:
#         model = Transaction
#         fields = [
#             'sender_name',
#             'beneficiary_name',
#             'beneficiary_address',
#             'beneficiary_phone',
#             'transaction_amount',
#             'created_by',
#             'destination_country',
#             'exchange_rate',
#         ]
#         widgets = {
#             'sender': forms.Select(attrs={'class': 'form-control'}),
#             'beneficiary_name': forms.TextInput(attrs={'class': 'form-control'}),
#             'beneficiary_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
#             'beneficiary_phone': forms.TextInput(attrs={'class': 'form-control'}),
#             'transaction_amount': forms.NumberInput(attrs={'class': 'form-control'}),
#             'created_by': forms.Select(attrs={'class': 'form-control'}),
#             'destination_country': forms.Select(attrs={'class': 'form-control'}),
#             'exchange_rate': forms.Select(attrs={'class': 'form-control'}),
#         }

#     def clean_transaction_amount(self):
#         amount = self.cleaned_data.get('transaction_amount')
#         if amount <= 0:
#             raise forms.ValidationError("Transaction amount must be greater than zero.")
#         return amount
#     class Meta:
#         model = Transaction
#         fields = [
#             'sender_name',
#             'beneficiary_name',
#             'beneficiary_address',
#             'beneficiary_phone',
#             'transaction_amount',
#             'destination_country',
#             'exchange_rate',
#         ]
#         widgets = {
#             'beneficiary_address': forms.Textarea(attrs={'rows': 3}),
#             'transaction_amount': forms.NumberInput(attrs={'step': '0.01'}),
#             'destination_country': forms.Select(),
#         }
#         labels = {
#             'sender_name': 'Sender',
#             'beneficiary_name': 'Beneficiary Name',
#             'beneficiary_address': 'Beneficiary Address',
#             'beneficiary_phone': 'Beneficiary Phone',
#             'transaction_amount': 'Amount to Send',
#             'destination_country': 'Destination Country',
#             'exchange_rate': 'Exchange Rate',
#         }

#     def __init__(self, *args, **kwargs):
#         user = kwargs.pop('user', None)
#         super(TransactionForm, self).__init__(*args, **kwargs)
#         if user:
#             self.instance.created_by = user


# class WalkinForm(forms.ModelForm):
#     class Meta:
#         model = Transaction
#         # fields = ["email","name",'phone']
#         fields =  ["sender_phone",'beneficiary_name',"beneficiary_address","beneficiary_phone","transaction_amount","destination_country"]
        

# class AccountHoldernForm(forms.ModelForm):
#     class Meta:
#         model = Transaction
#         # fields = ["email","name",'phone']
#         fields =  ["sender_phone",'beneficiary_name',"beneficiary_address","beneficiary_phone","transaction_amount","destination_country"]
        

# class PhoneValidation(forms.ModelForm):
#     class Meta:
#         model = Customer
#         fields = ["phone"]

#     def clean_phone(self):
#         phone = self.cleaned_data.get("phone")

#         # Run model-level validator
#         try:
#             validate_nigerian_prefix(phone)
#         except forms.ValidationError as e:
#             raise forms.ValidationError(f"Invalid phone number: {e.message}")

#         # Check for duplicates, excluding current instance (for updates)
#         if Customer.objects.filter(phone=phone).exclude(pk=self.instance.pk).exists():
#             raise forms.ValidationError("This phone number is already registered.")

#         return phone
    

# class CustomerForm(forms.ModelForm):
#     class Meta:
#         model = Customer
#         fields = ['name', 'phone', 'address', 'account_number', 'account_bal']
#         widgets = {
#             'address': forms.Textarea(attrs={'rows': 3}),
#         }

#     def clean_account_number(self):
#         account_number = self.cleaned_data['account_number']
#         if Customer.objects.filter(account_number=account_number).exists():
#             raise forms.ValidationError("🚫 This account number already exists.")
#         return account_number

#     def clean_phone(self):
#         phone = self.cleaned_data['phone']
#         # if Customer.objects.filter(phone=phone).exists():
#         #     raise forms.ValidationError("🚫 This phone number is already registered.")
#         return phone

# class TransactionWithAccountForm(forms.ModelForm):
#     class Meta:
#         model = Transaction
#         fields = [
#             "beneficiary_name", "beneficiary_phone",
#             "transaction_amount", "sender_address",
#             "beneficiary_address", "destination_country"
#         ]
#         widgets = {
#             "transaction_amount": forms.TextInput(attrs={
#                 "hx-post": reverse_lazy("balance_check"),
#                 "hx-trigger": "keyup",
#                 "hx-target": "#balance_check",
#                 # "autocomplete": "off",
#             }),
#         }

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.helper = FormHelper()
#         self.helper.form_method = 'POST'
#         # self.helper.add_input(Submit('submit', 'Look UP'))
#         self.helper.layout = Layout(
#             Field('beneficiary_name'),
#             Field('beneficiary_phone'),
#             Field('transaction_amount'),
#             HTML('<div class="text-danger mt-2" id="balance_check"></div>'),
#             Field('sender_address'),
#             Field('beneficiary_address'),
#             Field('destination_country'),
#             # HTML('<div class="custom-divider">-- Divider Between Names --</div>'),
#             # HTML('<div class="text-danger mt-2" id="balance_check2">Another div!</div>'),
#             # HTML('<p>More content inside the "ayo" div.</p>'),
#         )
# class TransactionWithoutAccountForm(forms.ModelForm):
#     class Meta:
#         model = Transaction
#         fields = [
#             "sender_name", "sender_phone","sender_address",
#             "beneficiary_name", "beneficiary_address", "beneficiary_phone",
#             "transaction_amount", "destination_country"
#         ]



# class CustomLoginForm(AuthenticationForm):
#     username = forms.CharField(label="Username", widget=forms.TextInput(attrs={'placeholder': 'Username'}))
#     password = forms.CharField(label="Password", widget=forms.PasswordInput(attrs={'placeholder': 'Password'}))


# class UserForm(forms.ModelForm):
#     class Meta:
#         model = User
#         fields = ['email', 'phone', 'full_name']

 
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.helper = FormHelper()
#         self.helper.layout = Layout(
#             Field('email'),
#     #     Fieldset(
#     #     'Tell us your favorite stuff {{ username }}',
#     #     'like_website',
#     #     'favorite_number',
#     #     'favorite_color',
#     #     'favorite_food',
#     #     HTML("""
#     #         <p>We use notes to get better, <strong>please help us {{ username }}</strong></p>
#     #     """),
#     #     'notes'
#     # ),
#             Field('phone'),
#             Field('full_name'),
#             Submit('submit', 'Submit', css_class='button white'),
#         )
 



# # https://gemini.google.com/app/2fd85cfc4be88c05

# class CripsyForm1b(forms.ModelForm):
#     class Meta:
#         model = Crispy
#         fields = ['name', 'phone']

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.helper = FormHelper()
#         self.helper.layout = Layout(
#             Field('name'),
#             # This is the custom div with id "ayo"
#             # HTML('<div id="ayo" class="my-custom-div-class">This is the extra div content!</div>'),
#              HTML('<div class="custom-divider">-- Divider Between Names --</div>'),
#              HTML('<p>More content inside the "ayo" div.</p>'),
#             Field('phone'),
#         )


# class RegistrationForm1(forms.ModelForm):
#     class Meta:
#         model = User
#         fields = ["email", "full_name" ]
    
 
# # https://django-crispy-forms.readthedocs.io/en/latest/layouts.html
# # {% load custom_tags %}


# class ExampleForm(forms.Form):

#     class Meta:
#         model = Crispy
#         fields = ['name', 'phone']
 
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.helper = FormHelper()
#         self.helper.layout = Layout(
#             Field('name'),
#     #     Fieldset(
#     #     'Tell us your favorite stuff {{ username }}',
#     #     'like_website',
#     #     'favorite_number',
#     #     'favorite_color',
#     #     'favorite_food',
#     #     HTML("""
#     #         <p>We use notes to get better, <strong>please help us {{ username }}</strong></p>
#     #     """),
#     #     'notes'
#     # ),
#             Field('phone'),
#             Submit('submit', 'Submit', css_class='button white'),
#         )