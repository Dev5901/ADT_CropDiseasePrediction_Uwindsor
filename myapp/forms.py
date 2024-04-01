from django import forms
from .models import User, UploadedImage

class SignUpForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password','password2']
        widgets = {
            'password': forms.PasswordInput()
        }

class LoginForm(forms.Form):
    model = User
    username = forms.CharField(max_length=50)
    password = forms.CharField(widget=forms.PasswordInput())


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image']
