from django import forms

class SignatureUploadForm(forms.Form):
    original_signature = forms.ImageField(label='Original Signature')
    test_signature = forms.ImageField(label='Test Signature')
