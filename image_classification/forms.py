from django import forms

from .models import Image


class UserImage(forms.ModelForm):
    class Meta:
        # To specify the model to be used to create form
        model = Image
        # It includes all the fields of model
        fields = ["image"]
