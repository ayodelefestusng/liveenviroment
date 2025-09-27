from django import forms
from .models import (
    Product,
    Category,
    PaymentMode,
    Customer,
    Refund,
    Supplier, # <-- Added Supplier to the imports
)


class ProductForm(forms.ModelForm):
    """
    A form for creating and updating Product instances.
    It includes fields for name, description, price, and category.
    The 'created_by' field is automatically handled in the view.
    """
    class Meta:
        model = Product
        # NOTE: Added 'supplier' to the fields list to match the form in create_product.html
        fields = ['name', 'description', 'price', 'image', 'category', 'supplier']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
        }


class CategoryForm(forms.ModelForm):
    """
    A form for creating and updating Category instances.
    """
    class Meta:
        model = Category
        fields = ['name']

from django import forms
from .models import PaymentMode

class SplitPaymentForm(forms.Form):
    """
    A dynamic form that creates one required field and other optional fields
    based on PaymentMode instances.
    """

    def __init__(self, *args, **kwargs):
        super(SplitPaymentForm, self).__init__(*args, **kwargs)
        payment_modes = PaymentMode.objects.all()

        for index, mode in enumerate(payment_modes):
            field_name = f"amount_{mode.id}"
            self.fields[field_name] = forms.DecimalField(
                label=f"{mode.name} Amount",
                max_digits=10,
                decimal_places=2,
                required=(index == 0)  # First one is required, rest are optional
            )
class PaymentModeForm(forms.ModelForm):
    """
    A form for creating and updating PaymentMode instances.
    """
    class Meta:
        model = PaymentMode
        fields = ['name']


class PaymentModeForm2(forms.ModelForm):
    """
    A form for creating and updating PaymentMode instances.
    The 'name' field is required only if no PaymentMode exists yet.
    """
    class Meta:
        model = PaymentMode
        fields = ['name']

    def __init__(self, *args, **kwargs):
        super(PaymentModeForm, self).__init__(*args, **kwargs)
        if PaymentMode.objects.exists():
            # Make 'name' optional if at least one PaymentMode exists
            self.fields['name'].required = False
        else:
            # First PaymentMode must have a name
            self.fields['name'].required = True


class CustomerForm(forms.ModelForm):
    """
    A form for creating and updating Customer instances.
    The 'user' field is automatically set to the logged-in user in the view.
    """
    class Meta:
        model = Customer
        fields = ['name', 'email', 'phone_number']


class RefundForm(forms.ModelForm):
    class Meta:
        model = Refund
        fields = ['order_item', 'comments', 'quantity']
        widgets = {
            'comments': forms.Textarea(attrs={'rows': 4}),
        }

    def __init__(self, *args, **kwargs):
        order = kwargs.pop('order', None)
        super().__init__(*args, **kwargs)
        if order:
            self.fields['order_item'].queryset = order.items.all()