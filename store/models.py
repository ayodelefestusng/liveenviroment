from django.db import models
from django.conf import settings
from django.utils.text import slugify
from decimal import Decimal
import uuid
from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver
from django.utils import timezone


class Customer(models.Model):
    # Changed user to be nullable to allow for anonymous customer creation via phone number
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='customer_profile')
    name = models.CharField(max_length=255, default="", null=True, blank=True)
    phone_number = models.CharField(max_length=15, unique=True)
    address = models.CharField(max_length=255, default="", null=True, blank=True)
    email = models.EmailField(default="", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name or self.phone_number


class Category(models.Model):
    name = models.CharField(max_length=200, unique=True)
    slug = models.SlugField(max_length=200, unique=True)

    class Meta:
        ordering = ['name']
        verbose_name_plural = 'Categories'

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class Supplier(models.Model):
    name = models.CharField(max_length=200, unique=True)
    contact_person = models.CharField(max_length=200, blank=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='suppliers_created')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Batch(models.Model):
    reference = models.CharField(max_length=50, unique=True, editable=False)
    supplier = models.ForeignKey(Supplier, on_delete=models.SET_NULL, null=True, related_name='batches')
    received_date = models.DateField(auto_now_add=True)
    expiry_date = models.DateField(null=True, blank=True)
    unit_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    invoice_id = models.CharField(max_length=255, default="N/A")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='batches_created')

    def save(self, *args, **kwargs):
        if not self.reference:
            self.reference = str(uuid.uuid4()).replace('-', '')[:10].upper()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.reference


class Product(models.Model):
    sku = models.CharField(max_length=100, unique=True, blank=True, null=True)
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    discount_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0.00)
    discount_absolute = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    image = models.ImageField(upload_to='products/', blank=True)
    
    # NOTE: Added the supplier field to resolve the inconsistency with forms.py
    supplier = models.ForeignKey(Supplier, on_delete=models.SET_NULL, null=True, related_name='products_supplied')

    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='products_created')
    edited_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='products_edited')

    class Meta:
        ordering = ['name']

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        if not self.sku:
            self.sku = str(uuid.uuid4()).replace('-', '')[:12].upper()
        super().save(*args, **kwargs)

    @property
    def final_price(self):
        if self.discount_percentage > 0:
            return self.price * (1 - self.discount_percentage / 100)
        elif self.discount_absolute > 0:
            return self.price - self.discount_absolute
        return self.price

    @property
    def has_discount(self):
        return self.discount_percentage > 0 or self.discount_absolute > 0

    @property
    def discount_amount(self):
        return self.price - self.final_price

    @property
    def discount_label(self):
        if self.discount_percentage > 0:
            return f"{self.discount_percentage}% off"
        elif self.discount_absolute > 0:
            return f"Save ₦{self.discount_absolute:.2f}"
        return ""

    def __str__(self):
        return self.name


class Inventory(models.Model):
    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name='inventory')
    batch = models.ForeignKey(Batch, on_delete=models.SET_NULL, null=True, related_name='inventory')
    quantity = models.PositiveIntegerField(default=0)
    min_quantity = models.PositiveIntegerField(default=10)
    unit_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    total_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='inventory_created')
    created_at = models.DateTimeField(default=timezone.now, null=False, blank=False)

    class Meta:
        verbose_name_plural = 'Inventory'

    @property
    def needs_reorder(self):
        return self.quantity <= self.min_quantity

    def __str__(self):
        return f"{self.product.name} ({self.quantity})"

@receiver(pre_save, sender=Inventory)
def calculate_inventory_costs(sender, instance, **kwargs):
    if instance.batch:
        instance.unit_cost = instance.batch.unit_cost
        instance.total_cost = instance.unit_cost * Decimal(instance.quantity)


class PaymentMode(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.name


class Order(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='store_orders' 
    )
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='customer_orders', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_paid = models.BooleanField(default=False)
    is_shipped = models.BooleanField(default=False)
    is_delivered = models.BooleanField(default=False)
    # The payment_mode field is no longer a single foreign key since we now support split payments.
    # It will be handled by the new Payment model.
    # payment_mode = models.ForeignKey(PaymentMode, on_delete=models.SET_NULL, null=True, blank=True)

    # Added new fields for enhanced checkout logic
    discount = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    final_amount = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    repayment_date = models.DateField(null=True, blank=True)
    
    @property
    def total_price(self):
        """Calculates the total price of the order before any discounts."""
        return sum(item.final_price for item in self.items.all())

    @property
    def total_quantity(self):
        """Calculates the total quantity of items in the order."""
        return sum(item.quantity for item in self.items.all())
    
    def __str__(self):
        return f"Order #{self.id}"


class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    
    @property
    def final_price(self):
        return self.product.final_price * self.quantity

    @property
    def profit(self):
        # This is a placeholder; you'd need a cost field on the Product or Inventory model
        return (self.product.final_price - self.product.price) * self.quantity

    def __str__(self):
        return f"{self.quantity} of {self.product.name}"


# New model to handle split payments
class Payment(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='payments')
    payment_mode = models.ForeignKey(PaymentMode, on_delete=models.SET_NULL, null=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Payment of ₦{self.amount} for Order #{self.order.id} via {self.payment_mode}"


class Refund(models.Model):
    order_item = models.ForeignKey(OrderItem, on_delete=models.CASCADE, related_name='refunds')
    quantity = models.PositiveIntegerField(default=1)
    comments = models.TextField(blank=True)
    time = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='refunds_created')

    def __str__(self):
        return f"Refund for Order Item #{self.order_item.id}"


class Beneficiary(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name


class Purpose(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name


class Expense(models.Model):
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    narrative = models.TextField()
    purpose = models.ForeignKey(Purpose, on_delete=models.SET_NULL, null=True, related_name='expenses')
    reference = models.CharField(max_length=255, unique=True, editable=False)
    beneficiary = models.ForeignKey(Beneficiary, on_delete=models.SET_NULL, null=True, related_name='expenses')
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='expenses_created')
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.reference:
            unique_part = str(uuid.uuid4()).replace('-', '')[:6].upper()
            purpose_name = self.purpose.name.replace(' ', '') if self.purpose else ''
            self.reference = f"{unique_part}/{purpose_name}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Expense: {self.reference}"


class RemoveItem(models.Model):
    batch = models.ForeignKey(Batch, on_delete=models.CASCADE, related_name='removed_items')
    inventory = models.ForeignKey(Inventory, on_delete=models.CASCADE, related_name='removed_items')
    quantity = models.PositiveIntegerField()
    comments = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='removed_items')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Removed {self.quantity} of {self.inventory.product.name} from Batch {self.batch.reference}"

@receiver(post_save, sender=Product)
def set_edited_by(sender, instance, created, **kwargs):
    # NOTE: This signal is currently a placeholder and needs to be implemented.
    if not created and instance.edited_by is None:
        pass
