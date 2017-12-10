# -*- coding: utf-8 -*-
from __future__ import (
    absolute_import, division, print_function, unicode_literals,
)

# Create your views here.
from django.contrib.auth import login
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.conf import settings

from .forms import LoginForm


def login_view(request):
    context = {}
    form = LoginForm(request.POST or None)
    context['form'] = form
    if request.POST and form.is_valid():
        if not form.is_valid():
            context['form_errors'] = str(form.errors['__all__'][0])
        else:
            user = form.authenticate(request)
            if user:
                login(request, user)
                return HttpResponseRedirect(settings.LOGIN_REDIRECT_URL)  # Redirect to a success page.

    return render(request, 'core/login.html', context)
