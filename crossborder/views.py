

# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
import random
import io

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils import timezone
from django.db import transaction as db_transaction
from decimal import Decimal
from datetime import timedelta
import random
from .models import Customer, Transaction, AccountTransactionT, TellerDetails
from .forms import TransactionWithAccountForm,TransactionWithoutAccountForm
from django.contrib.auth.decorators import login_required



from django.shortcuts import get_object_or_404
from django.http import HttpResponse, JsonResponse
from .models import Customer

from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from decimal import Decimal

from django.contrib.auth import  get_user_model
from .forms import CurencyForm,SenderAccountLookupForm,TransactionWithAccountForm
from .models import Customer,Transaction,TellerDetails,AccountTransactionT
User = get_user_model



# Create your views here.
def home  (request):
     return render(request, "transactions/ay.html")
    


@csrf_exempt
@login_required
def add_currency(request):
    if request.method == 'POST':
        form = CurencyForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('currency')  # Replace with your actual URL name
    else:
        form = CurencyForm()
    return render(request, 'transactions/operations.html', {'form': form})

@csrf_exempt
@login_required
def add_country(request):
    if request.method == 'POST':
        form = CountryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('country_list')  # Replace with your actual URL name
    else:
        forms = CountryForm()
    return render(request, 'transactions/operations.html', {'forms': forms})


@csrf_exempt
def check_account(request):

    if request.method == "GET":
        return HttpResponse("Oya")
    elif request.method == "POST":
        account_number = request.POST.get('account_number')
       
        

        if account_number and not Customer.objects.filter(account_number=account_number).exists():
            return HttpResponse("Invalid Account")

        customer = Customer.objects.get(account_number=account_number)
        request.session["sender_id"] = customer.phone
        print("AJADI", account_number)
        return redirect("transaction_with_account")
        # return HttpResponse("")  # Empty response if email is available or not provided


@login_required
def sender_with_account_lookup(request):
    form = SenderAccountLookupForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        account_number = form.cleaned_data["account_number"]

        try:
            customer = Customer.objects.get(account_number=account_number)
            request.session["sender_id"] = customer.phone

            if request.headers.get("HX-Request"):
                response = HttpResponse()
                response["HX-Redirect"] = reverse("transaction_with_account")
                return response
            else:
                return redirect("transaction_with_account")

        except Customer.DoesNotExist:
            if request.headers.get("HX-Request"):
                return HttpResponse(
                    '<div class="text-danger mt-2" id="account-err">❌ Invalid account!</div>',
                    status=200
                )
            else:
                form.add_error("account_number", "Invalid account")

    return render(request, "transactions/sender_with_account_lookup.html", {"form": form})


# @login_required
# def sender_with_account_lookup(request):
#     form = SenderAccountLookupForm(request.POST or None)

#     if request.method == "POST" and form.is_valid():
#         account_number = form.cleaned_data["account_number"]
#         try:
#             customer = Customer.objects.get(account_number=account_number)
#             request.session["sender_id"] = customer.phone

#             # HTMX redirect if triggered via HTMX
#             if request.headers.get("HX-Request"):
#                 response = HttpResponse()
#                 response["HX-Redirect"] = reverse("transaction_with_account")
#                 return response
#             else:
#                 return redirect("transaction_with_account")

#         except Customer.DoesNotExist:
#             # HTMX error response
#             if request.headers.get("HX-Request"):
#                 return HttpResponse('<div class="text-danger mt-2" id="account-err">Invalid account!</div>')
#                 return HttpResponse('<span class="text-danger">Invalid account</span>')
            
#             else:
#                 form.add_error("account_number", "Invalid account")

#     return render(request, "transactions/sender_with_account_lookup.html", {"form": form})



def balance_check(request):
    sender_id = request.session.get("sender_id")

    if request.method == "POST":
        try:
            total_amount = int(request.POST.get("transaction_amount", 0))
            
        except (ValueError, TypeError):
            return HttpResponse("Invalid transaction amount", status=400)

        customer = get_object_or_404(Customer, phone=sender_id)
        print ("Aluwe",customer.account_bal -total_amount)


        if customer.account_bal < total_amount:
            return HttpResponse("Insufficient balance", status=403)

        # If balance is sufficient, proceed with transaction logic
        return HttpResponse('<div class="text-success mt-2" id="balance-status">Balance is sufficient</div>', status=200)
        # return HttpResponse("Balance is sufficient", status=200)

    return HttpResponse("Invalid request method", status=405)
                
        



@csrf_exempt
@login_required

def is_duplicate_transaction(sender, beneficiary_phone, amount):
    time_threshold = timezone.now() - timedelta(minutes=5)
    return Transaction.objects.filter(
        sender=sender,
        beneficiary_phone=beneficiary_phone,
        transaction_amount=amount,
        created_at__gte=time_threshold
    ).exists()


@csrf_exempt
@login_required
def transaction_with_account1(request):
    # print("Ajauy",request.session.items())
   
    sender_id = request.session.get("sender_id")
    print("AjauyR",sender_id)
   
    if not sender_id:
        return redirect("sender_with_account_lookup")

    customer = get_object_or_404(Customer, phone=sender_id)
    
    form = TransactionWithAccountForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        
        transaction = form.save(commit=False)
        transaction.sender_name = customer.name
        transaction.sender_phone = customer.phone
        transaction.sender_address = customer.address
        transaction.created_by = request.user

        # Check for duplicate transaction within last 5 minutes
        time_threshold = timezone.now() - timedelta(minutes=5)
        duplicate_exists = Transaction.objects.filter(
            sender_name=customer.name,
            beneficiary_phone=transaction.beneficiary_phone,
            transaction_amount=transaction.transaction_amount,
            created_at__gte=time_threshold
        ).exists()

        if duplicate_exists:
            messages.warning(request, "⚠️ Duplicate transaction detected. Please wait before retrying.")
        else:
            # 💰 Calculate total amount
            transaction_fee = transaction.transaction_amount * Decimal("0.015")
            total_amount = transaction.transaction_amount + transaction_fee

            # 🔢 Generate 34-digit reference
            now_str = timezone.now().strftime("%Y%m%d%H%M%S")
            random_digits = str(random.randint(100000000000, 999999999999))
            trx_ref = f"{customer.account_number}{now_str}{random_digits}"[:34]

            # 🧾 Create AccountTransactionT entry
            teller_details = get_object_or_404(TellerDetails, user=request.user)
            AccountTransactionT.objects.create(
                reference=trx_ref,
                customer=customer,
                teller=teller_details,
                amount=total_amount
            )

            # 📉 Deduct from customer balance
            if customer.account_bal < total_amount:
                messages.error(request, "❌ Insufficient balance.")
                return render(request, "partial/transaction_with_account.html", {
                    "form": form,
                    "customer": customer
                })  

            customer.account_bal -= total_amount
            customer.save()

            # 📈 Increase branch till
            branch = teller_details.branch_code
            
            print ("Branch Till Balance Before", branch.branch_till_balance)
            branch.branch_till_balance += total_amount
            branch.save()
            print ("Branch Till Balance After", branch.branch_till_balance)

            # 💾 Save transaction
            transaction.save()
            messages.success(request, "✅ Transaction submitted successfully!")
            return redirect("sender_with_account_lookup")

    return render(request, "partial/transaction_with_account.html", {
        "form": form,
        "customer": customer
    })





@login_required
def transaction_with_account(request):
    sender_id = request.session.get("sender_id")
    print("arole",sender_id)
    if not sender_id:
        return redirect("sender_with_account_lookup")

    customer = get_object_or_404(Customer, phone=sender_id)
    form = TransactionWithAccountForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        print("arole2",sender_id)
        transaction = form.save(commit=False)
        transaction.sender_name = customer.name
        transaction.sender_phone = customer.phone
        transaction.sender_address = customer.address
        transaction.created_by = request.user

        # Check for duplicate transaction within last 5 minutes
        time_threshold = timezone.now() - timedelta(minutes=5)
        duplicate_exists = Transaction.objects.filter(
            sender_name=customer.name,
            beneficiary_phone=transaction.beneficiary_phone,
            transaction_amount=transaction.transaction_amount,
            created_at__gte=time_threshold
        ).exists()

        if duplicate_exists:
            messages.warning(request, "⚠️ Duplicate transaction detected. Please wait before retrying.")
        else:
            transaction_fee = transaction.transaction_amount * Decimal("0.015")
            total_amount = transaction.transaction_amount + transaction_fee

            now_str = timezone.now().strftime("%Y%m%d%H%M%S")
            random_digits = str(random.randint(100000000000, 999999999999))
            trx_ref = f"{customer.account_number}{now_str}{random_digits}"[:34]

            teller_details = get_object_or_404(TellerDetails, user=request.user)

            if customer.account_bal < total_amount:
                messages.error(request, "❌ Insufficient balance.")
                return render(request, "partial/transaction_with_account.html", {
                    "form": form,
                    "customer": customer
                })

            # Atomic transaction block
            with db_transaction.atomic():
                AccountTransactionT.objects.create(
                    reference=trx_ref,
                    customer=customer,
                    teller=teller_details,
                    amount=total_amount
                )

                customer.account_bal -= total_amount
                customer.save()

                branch = teller_details.branch_code
                branch.branch_till_balance += total_amount
                branch.save()

                transaction.save()

            messages.success(request, "✅ Transaction submitted successfully!")
            print("Bababa Ife", branch.branch_till_balance)
            return redirect("sender_with_account_lookup")

    return render(request, "partial/transaction_with_account.html", {
        "form": form,
        "customer": customer
    })




@csrf_exempt
@login_required
def transaction_without_account(request):
    print ("Ajadi4")
    form = TransactionWithoutAccountForm(request.POST or None)
    print ("Ajadi3")

    if request.method == "POST" and form.is_valid():
        transaction = form.save(commit=False)
        transaction.created_by = request.user
        print ("Ajadi2")

        # Check for duplicate transaction within last 5 minutes
        time_threshold = timezone.now() - timedelta(minutes=5)
        duplicate_exists = Transaction.objects.filter(
            sender_name=transaction.sender_name,
            sender_phone=transaction.sender_phone,
            beneficiary_phone=transaction.beneficiary_phone,
            transaction_amount=transaction.transaction_amount,
            created_at__gte=time_threshold
        ).exists()

        if duplicate_exists:
            messages.warning(request, "⚠️ Duplicate transaction detected. Please wait before retrying.")
        else:
            # 💰 Calculate total amount
            transaction_fee = transaction.transaction_amount * Decimal("0.015")
            total_amount = transaction.transaction_amount + transaction_fee

            # 🔢 Generate 34-digit reference
            now_str = timezone.now().strftime("%Y%m%d%H%M%S")
            random_digits = str(random.randint(100000000000, 999999999999))
            reference = f"{transaction.sender_phone}{now_str}{random_digits}"[:34]

            # 🧾 Get teller details
            teller_details = get_object_or_404(TellerDetails, user=request.user)

            # 💳 Check teller balance
            if teller_details.balance < total_amount:
                messages.error(request, "❌ Teller till has insufficient funds.")
                return render(request, "transactions/transaction_without_account.html", {"form": form})

            # 📉 Debit teller till
            print("Initial Balance",teller_details.balance)
            teller_details.balance -= total_amount

            print("Initial Balance",teller_details.balance)
            teller_details.save()
            print("Aftermat Balance",teller_details.balance)

            # 🧾 Create AccountTransactionT entry
            AccountTransactionT.objects.create(
                reference=reference,
                # customer=None,
                teller=teller_details,
                amount=total_amount
            )

            # 💾 Save transaction
            transaction.save()
            messages.success(request, "✅ Transaction submitted successfully!")
            return redirect("sender_with_account_lookup")

    return render(request, "transactions/transaction_without_account.html", {"form": form})





