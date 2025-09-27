from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import (
    Category, Supplier, Batch, Product, Inventory, Customer,
    Order, OrderItem, PaymentMode, Refund, Beneficiary, Purpose, Expense,
    RemoveItem, Payment
)


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug']
    prepopulated_fields = {'slug': ('name',)}


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'sku', 'price', 'category', 'created_by', 'edited_by']
    prepopulated_fields = {'slug': ('name',)}
    list_filter = ['category', 'created_by', 'edited_by']
    search_fields = ['name', 'sku']


@admin.register(Supplier)
class SupplierAdmin(admin.ModelAdmin):
    list_display = ['name', 'contact_person', 'phone_number', 'created_by']
    search_fields = ['name', 'contact_person']


@admin.register(Batch)
class BatchAdmin(admin.ModelAdmin):
    list_display = ['reference', 'supplier', 'received_date', 'expiry_date', 'invoice_id', 'created_by']
    list_filter = ['received_date', 'expiry_date', 'supplier']
    search_fields = ['reference', 'supplier__name']


@admin.register(Inventory)
class InventoryAdmin(admin.ModelAdmin):
    list_display = ['product', 'batch', 'quantity', 'unit_cost', 'total_cost', 'needs_reorder', 'created_by']
    list_filter = ['product', 'batch', 'created_by']
    search_fields = ['product__name', 'batch__reference']


@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ['user', 'name', 'phone_number', 'email', 'created_at']
    search_fields = ['user__email', 'name', 'phone_number']


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'customer', 'created_at', 'is_paid', 'is_shipped', 'is_delivered', 'get_payment_modes']
    list_filter = ['is_paid', 'is_shipped', 'is_delivered']
    search_fields = ['id', 'user__email', 'customer__phone_number']

    def get_payment_modes(self, obj):
        """Displays the payment modes for an order."""
        return ", ".join([payment.payment_mode.name for payment in obj.payments.all()])
    
    get_payment_modes.short_description = 'Payment Modes'


@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ['order', 'product', 'quantity', 'final_price', 'profit']
    list_filter = ['order']
    search_fields = ['product__name']


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['order', 'payment_mode', 'amount', 'timestamp']
    list_filter = ['payment_mode', 'timestamp']
    search_fields = ['order__id', 'payment_mode__name']


@admin.register(PaymentMode)
class PaymentModeAdmin(admin.ModelAdmin):
    list_display = ['name', 'description']


@admin.register(Refund)
class RefundAdmin(admin.ModelAdmin):
    list_display = ['id', 'order_item', 'quantity', 'time', 'created_by']
    list_filter = ['time', 'created_by']


@admin.register(Beneficiary)
class BeneficiaryAdmin(admin.ModelAdmin):
    list_display = ['name']
    search_fields = ['name']


@admin.register(Purpose)
class PurposeAdmin(admin.ModelAdmin):
    list_display = ['name']
    search_fields = ['name']


@admin.register(Expense)
class ExpenseAdmin(admin.ModelAdmin):
    list_display = ['reference', 'amount', 'purpose', 'beneficiary', 'created_by', 'created_at']
    list_filter = ['purpose', 'beneficiary', 'created_by']
    search_fields = ['reference', 'narrative', 'purpose__name', 'beneficiary__name']


@admin.register(RemoveItem)
class RemoveItemAdmin(admin.ModelAdmin):
    list_display = ['batch', 'inventory', 'quantity', 'created_by', 'created_at']
    list_filter = ['created_by', 'created_at']
    search_fields = ['batch__reference', 'inventory__product__name']
