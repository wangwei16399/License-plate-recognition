from django.shortcuts import render,redirect
from django.http import HttpResponse

def default(request):
    return redirect('/parkmanage/main')