from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from .models import Product, Category, Order, OrderItem, Customer, PaymentMode, Refund, Payment
from store.forms import CustomerForm, ProductForm, CategoryForm, PaymentModeForm
from django.views.decorators.http import require_POST
from django.contrib import messages
import uuid
import json
from decimal import Decimal

from decimal import Decimal, InvalidOperation
from django.views.decorators.http import require_POST
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseBadRequest
from .models import Order, Customer, Payment, PaymentMode

from django.shortcuts import get_object_or_404, render
from .models import PaymentMode

from django.http import HttpResponse
from .models import PaymentMode

@login_required
def home(request):
    """
    Renders the home page with a list of products.
    Handles product searching, category filtering, and sorting.
    """


    
    products = Product.objects.all()
    categories = Category.objects.all()

    active_order_id = request.session.get('active_order_id')
    active_order = Order.objects.filter(pk=active_order_id).first()

    search_query = request.GET.get('search')
    category_id = request.GET.get('category')
    sort_by = request.GET.get('sort_by')

    if search_query:
        products = products.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(sku__icontains=search_query)
        )

    if category_id:
        products = products.filter(category_id=category_id)

    if sort_by == 'price_asc':
        products = products.order_by('price')
    elif sort_by == 'price_desc':
        products = products.order_by('-price')
    elif sort_by == 'name_asc':
        products = products.order_by('name')
    elif sort_by == 'name_desc':
        products = products.order_by('-name')
    
    context = {
        'products': products,
        'categories': categories,
        'active_order': active_order
    }
    
    if request.htmx:
        return render(request, 'store/partials/product_list.html', context)
        
    return render(request, 'store/home.html', context)

@login_required
def product_detail(request, slug):
    """
    Renders the detail page for a single product.
    """
    product = get_object_or_404(Product, slug=slug)
    return render(request, 'store/product_detail.html', {'product': product})

@login_required
def order_list(request):
    """
    HTMX endpoint to get and render the list of a user's orders.
    """
    # order_id = request.session.get('active_order_id')
    # print ("Order Regg", order_id     ) 
    orders = Order.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'store/partials/order_list.html', {'orders': orders})

@login_required
@require_POST
def create_order(request):
    """
    HTMX endpoint to create a new, empty order for the user.
    """
    active_order_id = request.session.get('active_order_id')
    if active_order_id:
        try:
            active_order = Order.objects.get(id=active_order_id)
            active_order.is_active = False
            active_order.save()
        except Order.DoesNotExist:
            pass

    new_order = Order.objects.create(user=request.user, is_paid=False)
    request.session['active_order_id'] = str(new_order.id)
    
    active_order = new_order
    return render(request, 'store/partials/order_dashboard.html', {'active_order': active_order})

@login_required
def order_dashboard(request, pk):
    """
    HTMX endpoint to display the details of a specific order.
    """
    
    order = get_object_or_404(Order, pk=pk, user=request.user)
    request.session['active_order_id'] = str(order.id)
    # active_order_id = request.session['active_order_id'] 
    return render(request, 'store/partials/order_dashboard.html', {'active_order': order})

@login_required
@require_POST
def add_to_cart(request):
    """
    HTMX endpoint to add a product to the user's active cart.
    If no active cart exists, a new one is created.
    """
    product_id = request.POST.get('product_id')
    product = get_object_or_404(Product, id=product_id)

   

    # If the product is in the 'hotel' category, trigger customer registration
    if product.category.name.lower() == 'hotel':
        if not request.session.get('customer_registered'):
             initial_data = {'email': request.user.email} if hasattr(request.user, 'email') else {}
             form = CustomerForm(initial=initial_data)
             return render(request, 'store/partials/customer_registration_modal.html', {
        'form': form
    })
            # return render(request, 'store/partials/customer_registration_modal.html')
    
    order_id = request.session.get('active_order_id')
    if not order_id:
        active_order = Order.objects.create(user=request.user, is_paid=False)
        request.session['active_order_id'] = str(active_order.id)
    else:
        active_order = get_object_or_404(Order, id=order_id)
    

    order_item, created = OrderItem.objects.get_or_create(
        order=active_order,
        product=product
    )
    if not created:
        order_item.quantity += 1
        order_item.save()

    return render(request, 'store/partials/order_dashboard.html', {'active_order': active_order})

def view_cart(request):
    """
    Renders the cart dashboard partial.
    """
    order_id = request.session.get('active_order_id')
    active_order = None
    if order_id:
        try:
            active_order = Order.objects.get(id=order_id)
        except Order.DoesNotExist:
            pass
    return render(request, 'store/partials/order_dashboard.html', {'active_order': active_order})

@login_required
def get_edit_cart_item_form(request, product_id):
    """
    HTMX endpoint to get the form for editing a cart item's quantity.
    """
    if request.user.is_authenticated:
        
        order_id = request.session.get('active_order_id')
        print("Edit Cart Item Form Accessed", order_id)
        order = get_object_or_404(Order, id=order_id)
        product=get_object_or_404(Product, id=product_id)
        order_item = get_object_or_404(OrderItem, order=order, product=product)
        order_item_id = order_item.id  # Correctly get the order item ID

        context = {
            "order_item_id": order_item_id,
            'product_id': product_id,
            'current_quantity': order_item.quantity,
            'order_item': order_item, # Now correctly passing the order_item object
        }
        return render(request, 'store/partials/edit_cart_item_form.html', context)
    else:
        return HttpResponse(status=401)

@require_POST
def edit_cart_item(request, product_id):
    """
    Edits the quantity of a cart item.
    """
    quantity = int(request.POST.get('quantity'))
    order_id = request.session.get('active_order_id')
    order = get_object_or_404(Order, id=order_id)
    product=get_object_or_404(Product, id=product_id)

    if order:
        order_item = get_object_or_404(OrderItem, order=order, product=product)
        order_item.quantity = quantity
        order_item.save()
        print("Order Item Updated:", order_item)
        # return redirect('store:order-list')
        # return redirect('store:home')
    
    return render(request, 'store/partials/order_dashboard.html', {'active_order': order})

@login_required
def customer_registration(request):
    """
    Handles customer registration via a modal.
    """
    if request.method == 'POST':
        form = CustomerForm(request.POST)
        if form.is_valid():
            customer = form.save(commit=False)
            customer.user = request.user
            customer.save()
            return HttpResponse(status=204, headers={'HX-Trigger': 'refresh-dashboard'})
    else:
        initial_data = {'email': request.user.email} if hasattr(request.user, 'email') else {}
        form = CustomerForm(initial=initial_data)

    return render(request, 'store/partials/customer_registration_modal.html', {
        'form': form
    })

def create_product(request):
    """
    Handles the creation of a new product.
    """
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            product = form.save(commit=False)
            product.created_by = request.user
            product.save()
            messages.success(request, "Product created successfully!")
            return redirect('store:operational')
    else:
        form = ProductForm()
    
    return render(request, 'store/create_product.html', {'form': form})

def operational_dashboard(request):
    """
    Renders the operational dashboard for store management.
    """
    return render(request, 'store/operational.html', {})

def create_category(request):
    """
    Handles the creation of a new category.
    """
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Category created successfully!")
            return redirect('store:operational')
    else:
        form = CategoryForm()
    
    return render(request, 'store/create_category.html', {'form': form})

def create_payment_mode(request):
    """
    Handles the creation of a new payment mode.
    """

    if request.method == 'POST':
        form = PaymentModeForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Payment mode created successfully!")
            return redirect('store:operational')
    else:
        form = PaymentModeForm()
    
    return render(request, 'store/create_payment_mode.html', {'form': form})

@require_POST
def remove_from_cart(request, product_id):
    """
    Removes a product from the user's active order.
    """
    order_id = request.session.get('active_order_id')
    if order_id:
        try:
            order = Order.objects.get(id=order_id)
            order_item = get_object_or_404(OrderItem, order=order, product__id=product_id)
            order_item.delete()
        except Order.DoesNotExist:
            messages.error(request, "Active order not found.")
    active_order = Order.objects.filter(id=order_id).first()
    return render(request, 'store/partials/order_dashboard.html', {'active_order': active_order})

def checkout(request):
    """
    Renders the checkout page.
    """
    
    order_id = request.session.get('active_order_id')
    if not order_id:
        print("Checkout View Accessed")
        return redirect('store:home')
    
    active_order = get_object_or_404(Order, pk=order_id)
    payment_modes = PaymentMode.objects.all()
    
    context = {
        'active_order': active_order,
        'payment_modes': payment_modes,
        'total_quantity': active_order.total_quantity,
        'total_amount': active_order.total_price,
    }
    
    if request.htmx:
        print("HTMX Request for Checkout Summary")
        return render(request, 'store/partials/checkout_summary.html', context)
    else:
        return render(request, 'store/checkout_page.html', context)

@require_POST
def check_customer_by_phone(request):
    print("Check Customer by Phone Accessed")
    """
    HTMX endpoint to check if a customer exists by phone number.
    Returns customer info or a message if not found.
    """
    phone_number = request.POST.get('phone_number')
    if len(phone_number) < 11:
        pass
    
    else:
        if not phone_number or len(phone_number) != 11 or not phone_number.startswith(('070', '080', '081', '090')):
            return HttpResponse('<div class="text-danger mt-2">Invalid phone number format.</div>')

    try:
        customer = Customer.objects.get(phone_number=phone_number)
        return render(request, 'store/partials/customer_info.html', {'customer': customer})
    except Customer.DoesNotExist:
        return HttpResponse('<div class="text-warning mt-2">Customer not registered. A new profile will be created.</div>')


def refund_order_items(request, pk):
    """
    HTMX endpoint to display and handle refund form for a specific order.
    """
    pass

def check_credit_mode(request, mode_id):
    mode = get_object_or_404(PaymentMode, pk=mode_id)
    if mode.name.lower() == 'credit':
        return render(request, 'store/partials/repayment_date_field.html')
    return render(request, 'store/partials/empty.html')  # return blank if not credit


@require_POST
def complete_order(request):
    """
    HTMX endpoint to finalize and complete an order.
    Validates split payments and handles credit repayment.
    """
    order_id = request.session.get('active_order_id')
    active_order = get_object_or_404(Order, pk=order_id)

    phone_number = request.POST.get('phone_number')
    discount_raw = request.POST.get('discount_amount', '0')

    try:
        discount_amount = Decimal(discount_raw)
    except (InvalidOperation, TypeError):
        return HttpResponseBadRequest("Invalid discount amount.")

    payment_mode_ids = request.POST.getlist('payment_modes')
    if not payment_mode_ids:
        return HttpResponseBadRequest("At least one payment mode must be selected.")

    # Get or create customer
    customer, _ = Customer.objects.get_or_create(phone_number=phone_number)

    # Apply discount
    net_amount = active_order.total_price - discount_amount

    # Collect and validate split payments
    total_paid = Decimal('0.00')
    payments = []

    for payment_mode_id in payment_mode_ids:
        amount_key = f'amount_{payment_mode_id}'
        try:
            amount_paid = Decimal(request.POST.get(amount_key, '0'))
        except (InvalidOperation, TypeError):
            return HttpResponseBadRequest("Invalid payment amount.")

        if amount_paid <= 0:
            return HttpResponseBadRequest("Each payment must be greater than zero.")

        payments.append((payment_mode_id, amount_paid))
        total_paid += amount_paid

    # Validate total paid matches net amount
    if total_paid != net_amount:
        return HttpResponseBadRequest("Split payments must match the total order amount after discount.")

    # Save order details
    active_order.customer = customer
    active_order.discount = discount_amount
    active_order.is_paid = True

    # Save payments
    for payment_mode_id, amount_paid in payments:
        Payment.objects.create(
            order=active_order,
            payment_mode_id=payment_mode_id,
            amount=amount_paid
        )

        # Handle credit repayment date
        mode = PaymentMode.objects.get(pk=payment_mode_id)
        if mode.name.lower() == 'credit':
            repayment_date = request.POST.get('repayment_date')
            active_order.repayment_date = repayment_date

    active_order.save()

    # Clear active order from session
    del request.session['active_order_id']

    return render(request, 'store/partials/order_success.html', {'active_order': active_order})

def toggle_credit(request):
    selected_ids = request.POST.getlist('payment_modes')
    credit_mode = PaymentMode.objects.filter(name__iexact='credit').first()

    if credit_mode and str(credit_mode.id) in selected_ids:
        return render(request, 'store/partials/repayment_date_field.html')
    return HttpResponse("")  # Return empty to hide the field



from django.shortcuts import get_object_or_404, redirect, render
from django.contrib import messages
from django.db import transaction
from .models import Order, OrderItem, Refund, Inventory
from .forms import RefundForm
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse
from django.contrib import messages
from django.db import transaction
from .models import Order, OrderItem, Refund, Inventory
from .forms import RefundForm


from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages
from django.db import transaction
from .models import Order, Refund
from .forms import RefundForm

from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages
from django.db import transaction
from .models import Order, Refund
from .forms import RefundForm

def refund_order_items(request, pk):
    order = get_object_or_404(Order, pk=pk)
    order_items = order.items.all()

    # Step 3: Block discounted orders
    if order.discount > 0:
        messages.error(request, "Refund not allowed: This order has a discount applied.")
        return redirect('store:order-dashboard', pk=pk)

    if request.method == 'POST':
        form = RefundForm(request.POST, order=order)
        if form.is_valid():
            refund = form.save(commit=False)
            order_item = refund.order_item

            # Step 2: Validate return quantity
            refunded_qty = sum(r.quantity for r in order_item.refunds.all())
            available_qty = order_item.quantity - refunded_qty

            if refund.quantity > available_qty:
                messages.error(request, f"Refund exceeds available quantity. Only {available_qty} item(s) can be refunded.")
                return redirect('store:order-dashboard', pk=pk)

            with transaction.atomic():
                # Step 4: Update inventory
                inventory = order_item.product.inventory
                inventory.quantity += refund.quantity
                inventory.save()

                refund.created_by = request.user
                refund.save()

            messages.success(request, f"Refund of {refund.quantity} item(s) processed successfully.")
            return redirect('store:order-dashboard', pk=pk)
    else:
        form = RefundForm(order=order)

    return render(request, 'store/refund_form.html', {
        'order': order,
        'order_items': order_items,
        'form': form
    })

def submit_refund(request):
    if request.method != 'POST':
        return HttpResponse(status=405)  # Method not allowed

    order_id = request.POST.get('order_id')
    if not order_id:
        return HttpResponse("Missing order ID", status=400)

    order = get_object_or_404(Order, pk=order_id)
    form = RefundForm(request.POST)
    form.fields['order_item'].queryset = order.items.all()

    if order.discount > 0:
        return render(request, 'store/partials/refund_blocked.html', {'order': order})

    if form.is_valid():
        refund = form.save(commit=False)
        order_item = refund.order_item
        print("Refund Requested for Order Item:", order_item)

        refunded_qty = sum(r.quantity for r in order_item.refunds.all())
        available_qty = order_item.quantity - refunded_qty

        if refund.quantity > available_qty:
            return render(request, 'store/partials/refund_error.html', {
                'error': f"Refund exceeds available quantity. Only {available_qty} item(s) can be refunded.",
                'form': form,
                'order': order
            })
        


        with transaction.atomic():
            inventory, created = Inventory.objects.get_or_create(product=order_item.product)
            inventory.quantity += refund.quantity
            inventory.save()

            refund.created_by = request.user
            refund.save()

        return render(request, 'store/partials/refund_success.html', {
            'refund': refund,
            'order': order
        })

    return render(request, 'store/partials/refund_error.html', {
        'error': "Invalid form submission.",
        'form': form,
        'order': order
    })


def create_refund(request):
    order_id = request.GET.get('order_id')
    order = get_object_or_404(Order, pk=order_id)
    order_items = order.items.all()
    print("Create Refund Accessed for Order:", order_id)

    if order.discount > 0:
        return render(request, 'store/partials/refund_blocked.html', {'order': order})

    return render(request, 'store/partials/refund_items.html', {
        'order': order,
        'order_items': order_items,
        'form': RefundForm()
    })
