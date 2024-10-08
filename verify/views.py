from django.shortcuts import render
from .forms import SignatureUploadForm
from tensorflow.keras.preprocessing import image
import numpy as np
from .models import create_signature_verification_model
from io import BytesIO
from PIL import Image
import os

# Function to preprocess the signature image
def preprocess_signature(image_file):
    img = Image.open(image_file)  # Open image using PIL
    img = img.resize((64, 64))  # Resize the image to the required target size
    img = img.convert('L')  # Convert the image to grayscale
    
    img_array = image.img_to_array(img)  # Convert to array (64, 64, 1)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale (64, 64, 1)
    img_array /= 255.0  # Normalize the image

    return img_array

# Path to the saved model weights
BASE_DIR = os.path.dirname(__file__)
k = os.path.join(BASE_DIR, 'siamese_net.h5')

# View for verifying signatures
def verify_signature(request):
    if request.method == 'POST':
        form = SignatureUploadForm(request.POST, request.FILES)
        if form.is_valid():
            original_signature = form.cleaned_data['original_signature']
            test_signature = form.cleaned_data['test_signature']

            # Preprocess the images
            original_img = preprocess_signature(original_signature)
            test_img = preprocess_signature(test_signature)

            # Expand dimensions to match model input shape (1, 64, 64, 1)
            original_img = np.expand_dims(original_img, axis=0)
            test_img = np.expand_dims(test_img, axis=0)

            # Load the trained model and its weights
            model = create_signature_verification_model()
            model.load_weights(k)  # Load the trained model weights

            # Make prediction
            result = model.predict([original_img, test_img])

            # Decide if the signatures match
            match = result > 0.5  # Threshold to determine if signatures match

            return render(request, 'upload.html', {'match': match})
    else:
        form = SignatureUploadForm()

    return render(request, 'upload.html', {'form': form})
