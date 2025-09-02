from django.shortcuts import render


# Create your views here.
from django.http import HttpResponse

def homes(request):
    return HttpResponse("iiid")

    