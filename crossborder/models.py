from django.db import models
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
from decimal import Decimal

from django.contrib.auth import get_user_model
User =get_user_model()
# Create your models here.

class Currency(models.Model):
    name = models.CharField(max_length=50, unique=True)
    rate = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"{self.name}"
    


class Country(models.Model):
    name = models.CharField(max_length=100, unique=True)
    currency = models.ForeignKey(Currency, on_delete=models.SET_NULL, null=True, related_name='countries')

    def __str__(self):
        return self.name


class Client(models.Model):
    bank_name = models.CharField(max_length=100, help_text="Full name of the client",unique=True)
    country = models.ForeignKey( Country, on_delete=models.SET_NULL, null=True, help_text="Country associated with the client")

    def __str__(self):
        return f"{self.bank_name} ({self.country.name if self.country else 'No Country'})"





def validate_branch_till(value):
    if not value.isdigit():
        raise ValidationError("Branch till must contain only digits.")
    if len(value) != 10:
        raise ValidationError("Branch till must be exactly 10 digits.")

class BranchDetails(models.Model):
    branch_code = models.CharField(max_length=3,validators=[RegexValidator(regex=r'^\d{3}$', message="Branch code must be exactly 3 digits.")],
        unique=True, help_text="Exactly 3-digit numeric code for the branch")
    
    branch_name = models.CharField(max_length=100, help_text="Full name of the branch" )
    bank_name = models.ForeignKey(Client, on_delete=models.SET_NULL,null=True,help_text="Bank associated with this branch")
    branch_till = models.CharField( max_length=10, unique=True, validators=[validate_branch_till], help_text="10-digit Branch Till account number")
    branch_till_balance = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)

    def clean(self):
        super().clean()
        if self.branch_till and self.branch_code:
            expected_prefix = self.branch_code
            if not self.branch_till.startswith(expected_prefix):
                raise ValidationError({
                    'branch_till': f"Branch till must start with branch code '{expected_prefix}'."
                })
            if len(self.branch_till) != 10:
                raise ValidationError({
                    'branch_till': "Branch till must be exactly 10 digits."
                })

    def save(self, *args, **kwargs):
        # Auto-generate branch_till if not set
        if not self.branch_till and self.branch_code:
            self.branch_till = f"{self.branch_code}0001111"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.branch_name} ({self.branch_code})"
    

def validate_teller_till(value):
    if not value.isdigit():
        raise ValidationError("Teller till must contain only digits.")
    if len(value) != 10:
        raise ValidationError("Teller till must be exactly 10 digits.")

class TellerDetails(models.Model):
    branch_code = models.ForeignKey( BranchDetails,on_delete=models.CASCADE,help_text="Branch where the teller is assigned")
    teller_till = models.CharField(max_length=10,unique=True,validators=[validate_teller_till],help_text="10-digit teller account number")
    user = models.ForeignKey(User,on_delete=models.CASCADE,help_text="User associated with this teller")
    balance = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=Decimal("0.00"),
        help_text="Current balance in teller's till"
    )

    

    def clean(self):
        super().clean()
        if self.teller_till and self.branch_code:
            expected_prefix = f"{self.branch_code.branch_code}7777"
            if not self.teller_till.startswith(expected_prefix):
                raise ValidationError({'teller_till': f"Teller till must start with '{expected_prefix}' followed by 2 digits."})
            if len(self.teller_till) != 10:
                raise ValidationError({'teller_till': "Teller till must be exactly 10 digits."})

    def save(self, *args, **kwargs):
        if not self.teller_till and self.branch_code:
            prefix = f"{self.branch_code.branch_code}7777"

            # Get existing tellers with same prefix
            existing = TellerDetails.objects.filter(teller_till__startswith=prefix).order_by("teller_till")

            # Extract last two digits and increment
            used_numbers = [
                int(t.teller_till[-2:]) for t in existing if t.teller_till[-2:].isdigit()
            ]
            next_number = max(used_numbers, default=0) + 1

            if next_number > 99:
                raise ValidationError("Maximum number of tellers reached for this branch.")

            self.teller_till = f"{prefix}{next_number:02d}"

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Teller {self.user.username} - {self.teller_till} ({self.branch_code.branch_name})"





# ✅ Valid Nigerian phone prefixes (4-digit only)
VALID_PREFIXES = {
    '0809', '0817', '0818', '0909', '0908',  # 9mobile
    '0701', '0708', '0802', '0808', '0812', '0901', '0902', '0904', '0907', '0912', '0911',  # Airtel
    '0705', '0805', '0807', '0811', '0815', '0905', '0915',  # Glo
    '0804',  # Mtel
    '0703', '0706', '0803', '0806', '0810', '0813', '0814', '0816', '0903', '0906', '0913', '0916', '0704', '0707'  # MTN
}

# 📞 Validator for Nigerian phone prefixes
def validate_nigerian_prefix(value):
    if not value.isdigit():
        raise ValidationError("Phone number must contain only digits.")
    if len(value) != 11:
        raise ValidationError("Phone number must be exactly 11 digits.")
    if value[:4] not in VALID_PREFIXES:
        raise ValidationError(f"Phone number must start with a valid Nigerian prefix. Got '{value[:4]}'.")


# 🔢 Validator for 10-digit account number
def validate_account_number(value):
    if not value.isdigit():
        raise ValidationError("Account number must contain only digits.")
    if len(value) != 10:
        raise ValidationError("Account number must be exactly 10 digits.")

# 🧾 Customer model
class Customer(models.Model):
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=20, validators=[validate_nigerian_prefix])
    address = models.TextField()
    account_number = models.CharField(max_length=10, unique=True, validators=[validate_account_number])
    account_bal = models.DecimalField(max_digits=12, decimal_places=2, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)  # 👈 New field


    def __str__(self):
        return self.name



# def generate_reference():
#     characters = string.ascii_uppercase + string.digits
#     return ''.join(random.choices(characters, k=13))
def generate_reference():
    import uuid
    return uuid.uuid4().hex[:13].upper()


class Transaction(models.Model):
    trx_code = models.CharField(max_length=13, unique=True, editable=False, default='', help_text="Auto-generated 13-character alphanumeric reference" )
    # sender = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='transactions')
    sender_name = models.CharField(max_length=100)

    sender_phone = models.CharField(max_length=11,validators=[validate_nigerian_prefix], help_text="11 digits and  a valid phone number format")
    beneficiary_name = models.CharField(max_length=100)
    beneficiary_address = models.TextField()
    sender_address = models.TextField()
    beneficiary_phone = models.CharField( max_length=11,validators=[validate_nigerian_prefix], help_text="11 digits and  a valid phone number format")
    transaction_amount = models.DecimalField(max_digits=12, decimal_places=2)
    trx_ref = models.CharField(max_length=100) 
    destination_country = models.ForeignKey(Country, on_delete=models.SET_NULL, null=True)

    transaction_fee = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True, default=0.00)
    total_amount = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True, default=0.00)
    exchange_rate = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True, default=0.00)
    receive_amount = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True, default=0.00)
    transaction_status = models.BooleanField( default=False,help_text="False = Pending, True = Completed")


    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.trx_code:
            self.trx_code = generate_reference()

        self.transaction_fee = self.transaction_amount * Decimal('0.015')
        self.total_amount = self.transaction_amount + self.transaction_fee

        currency = self.destination_country.currency if self.destination_country else None
        rate = currency.rate if currency and currency.rate > 0 else None
        self.exchange_rate = rate

        self.receive_amount = ( self.transaction_amount / rate if rate else Decimal('0.00'))

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Transaction from {self.sender_name} to {self.beneficiary_name} with{self.trx_code} "
    


class BranchAccountTill(models.Model):
    TRANSACTION_TYPES = [
        ('Customer', 'Customer'),
        ('Teller', 'Teller'),
    ]

    account_number = models.CharField(
        max_length=10,
        unique=True,
        help_text="10-digit branch account number"
    )
    branch_code = models.ForeignKey(
        BranchDetails,
        on_delete=models.CASCADE,
        help_text="Branch associated with this account"
    )
    total_amount = models.ForeignKey(
        Transaction,
        on_delete=models.CASCADE,
        related_name='till_amount',
        help_text="Transaction amount affecting the till"
    )
    transaction_type = models.CharField(
        max_length=10,
        choices=TRANSACTION_TYPES,
        help_text="Type of transaction: Customer or Teller"
    )
    account_balance = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0.00,
        help_text="Current balance of the account"
    )
    reference = models.ForeignKey(
        "Transaction",
        on_delete=models.CASCADE,
        related_name='till_reference',
        help_text="Reference to the transaction"
    )
    posted_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        help_text="User who posted the transaction"
    )

    def save(self, *args, **kwargs):
        # Automatically update account balance when saving
        if self.total_amount:
            self.account_balance += self.total_amount.amount  # assuming Transaction has an 'amount' field
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.account_number} ({self.branch_code.branch_name}) - Balance: ₦{self.account_balance}"



def get_default_customer():
    return Customer.objects.first().id  # Or any logic you prefer


class AccountTransactionT(models.Model):
    reference = models.CharField(max_length=50, unique=True)
    # customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='transactions',default=1)
    customer = models.ForeignKey(
    Customer,
    on_delete=models.CASCADE,
    related_name='transactions',
    default=get_default_customer
)
    teller = models.ForeignKey(TellerDetails, on_delete=models.CASCADE, related_name='transactions')
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    timestamp = models.DateTimeField(auto_now_add=True)



# https://gemini.google.com/app/2fd85cfc4be88c05
class Crispy(models.Model):
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=20)
    def __str__(self):
        return self.name