from django import template

register = template.Library()

@register.filter
def get_field(form, field_name):
    """Get form field by name"""
    return form[field_name]