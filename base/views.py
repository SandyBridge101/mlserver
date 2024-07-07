from django.shortcuts import render
from django.http import HttpResponse,JsonResponse

from .main_code.main import get_recommendations

# Create your views here.

def modelResponseView(request,entry):#If bank in added:add expiry date and card number. the pin corresponds to the momo 4-digit and bank 3-digit
    results,message=get_recommendations(entry)

    response_data = {
        'input':entry,
        'recommendations_list': results,
        'message': message
    }

    return JsonResponse(response_data)