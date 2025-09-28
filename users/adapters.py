from allauth.socialaccount.adapter import DefaultSocialAccountAdapter

class MySocialAccountAdapter(DefaultSocialAccountAdapter):
    def populate_user(self, request, sociallogin, data):
        user = super().populate_user(request, sociallogin, data)
        user.full_name = data.get('name', '')  # Google returns 'name' as full name
        return user

    def is_open_for_signup(self, request, sociallogin):
        # Allow auto signup only if full_name is present
        name = sociallogin.account.extra_data.get('name')
        return bool(name)