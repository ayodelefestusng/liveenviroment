from django.http import HttpResponseNotFound

class DomainRouterMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        host = request.get_host().split(':')[0]

        # Normalize localhost variants
        if host in ['localhost', '127.0.0.1']:
            request.urlconf = 'users.urls'
        elif host == 'buyrite.ng':
            request.urlconf = 'buyrite.urls'
        elif host == 'kunle.com':
            request.urlconf = 'store.urls'
        else:
            return HttpResponseNotFound("Unknown domain")

        return self.get_response(request)