from django.contrib.auth.hashers import make_password
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout

from django.urls import reverse
import os
from .forms import ImageUploadForm, SignUpForm, LoginForm
# from .models import User
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import check_password
from myapp.models import User, UploadedImage
# from .models import ImageModel  # Assuming ImageModel is your model for MongoDB images
#from .ml_utils import load_custom_model, preprocess_image

#importing packages
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def home(request):
    return render(request, 'home.html')

def user_signup_login(request): 
    if request.method == 'POST':
       value = request.POST.get('custom')
       
       if value == 'signup':
           # Assuming 'request' is the HttpRequest object and 'password' is the password from request.POST
           raw_password = request.POST.get('password')

           # Hash the raw password
           hashed_password = make_password(raw_password)
           mutable_post = request.POST.copy()

           # Update the password field in the mutable copy
           mutable_post['password'] = hashed_password

           form = SignUpForm(mutable_post)
           if form.is_valid():
              user = form.save()
              login(request, user)
              return render(request, 'SignupLogin.html')
           else:  
              return render(request, 'SignupLogin.html', {'form': form})

       if value == 'login':
           form = LoginForm(request.POST)
           if form.is_valid():
                username = request.POST.get('username')
                password = request.POST.get('password')

                user = authenticate_user(username=username, password=password)
                if user is not None:
                    login(request, user)
                    return redirect('image_upload')
                else:
                    # Handle invalid login
                    return render(request, 'SignupLogin.html', {'form': form})

    else:   
        return render(request, 'SignupLogin.html')

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        print("data: ",request.POST)
        print("valid: ",form.is_valid())
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('image_upload')
    else:
        form = AuthenticationForm()
        return render(request, 'login.html', {'form': form})

def user_signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('login')
    else:
        form = UserCreationForm()
        return render(request, 'signup.html', {'form': form})

def image_upload(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the uploaded image here
            instance = form.save()
            # get the image path from the saved instance
            image_path = instance.image.url
            print("IMAGE PATH: ", image_path)
            absolute_path = "C:/Users/Dev Prajapati/Desktop/ADTProject/prjct 3" + image_path
            final_path = replace_slash(absolute_path)
            print("ABSOLUTE PATH: ", absolute_path)
            # Brown percentage, predicted label
            result = analyze_image(absolute_path)
            #result = analyze_image("C:/Users/Dev Prajapati/PycharmProjects/ADTProject/dataset/spilt_Dataset/test/Healthy/Corn_Health (12).jpg")
            severity = result['brown_percentage']
            print("SEVERITY: ", severity)
            sev = str(severity)[:5]
            predicted_label = result['predicted_label']
            suggestion = suggest_management(predicted_label, severity)
            context = {
            'suggestion': suggestion,
            'sev': sev,
            'predicted_label': predicted_label,
            }
            return render(request, 'image_upload_success.html', context)
            #return render(request, 'image_upload_success.html', {'suggestion': suggestion}, {'severity': severity})
            #return redirect(reverse('image_upload_success') + '?suggestion='suggestion)
    else:
        form = ImageUploadForm()
    # Pass the form and suggestion variable to the template context
    suggestion = request.GET.get('suggestion', '')  # Get suggestion from query parameters
    return render(request, 'image_upload.html', {'form': form, 'suggestion': suggestion})

def replace_slash(slash_path):
    return slash_path.replace('\\', '/')

def analyze_image(image_path):
    # Load the trained InceptionV3 model
    model = load_model('myapp/inceptionV3_model.h5')  # Provide the path to your saved model

    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to the range [0, 1]

    # Make predictions
    predictions = model.predict(img_array)

    # Get the class label with the highest probability
    predicted_class = np.argmax(predictions[0])

    # Map class indices to class labels (assuming classes are labeled as 0, 1, 2, 3)
    class_labels = {0: 'Blight', 1: 'Common Rust', 2: 'Gray Leaf Spot', 3: 'Healthy'}

    # Get the predicted class label
    predicted_label = class_labels[predicted_class]

    # Read the image using OpenCV
    image_result = cv2.imread(image_path)

    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(image_result, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the brown color in HSV
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([15, 255, 255])

    # Threshold the HSV image to get only brown colors
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Calculate the total number of pixels in the image
    total_pixels = np.prod(image_result.shape[:2])

    # Calculate the number of brown pixels
    brown_pixels = cv2.countNonZero(brown_mask)

    # Calculate the brown percentage
    brown_percentage = (brown_pixels / total_pixels) * 100

    return {'brown_percentage': brown_percentage, 'predicted_label': predicted_label}


def convert_windows_path_to_python_macos(windows_path):
    # Replace backslashes with forward slashes
    python_path = windows_path.replace('\\', '/')
    
    # Convert drive letter to lowercase for macOS compatibility
    if ':' in python_path:
        drive_letter, rest_of_path = python_path.split(':', 1)
        python_path = '/' + drive_letter.lower() + rest_of_path
    
    return python_path



def image_upload_success(request):#, image_path):
    if  request.method == 'GET':
        return render(request, 'image_upload_success.html')

    else:
        return redirect('image_upload')


def logout_view(request):
    logout(request)
    # Redirect to a specific URL after logout
    return redirect('home')    


def authenticate_user(username, password):
    try:
        # Retrieve the user object from the database
        user = User.objects.get(username=username)
        # Check if the password matches using Django's check_password function
        # Get the password from the user object
        user_password = user.get_password()
        if check_password(password, user.password):
            return user  # Authentication successful
        else:
            return None  # Password does not match
    except User.DoesNotExist:
        return None  # User does not exist

def suggest_management(category, severity):
    if category == "Healthy":
        return "Continue regular maintenance such as watering, fertilizing, and monitoring for pests and diseases."
    elif category == "Common Rust":
        if severity <= 10:
            return "Monitor plants regularly for signs of rust. Remove and destroy infected leaves to prevent spread."
        elif severity <= 40:
            return "Apply fungicides labeled for rust control following manufacturer's instructions. Consider cultural practices such as spacing plants to improve air circulation."
        elif severity <= 70:
            return "Increase frequency of fungicide applications. Prune heavily infected parts of plants and dispose of them properly."
        else:
            return "Drastically reduce the spread by removing severely infected plants and implementing strict sanitation practices. Consider crop rotation for future seasons."
    
    elif category == "Blight":
        if severity <= 10:
            return "Monitor plants for early signs of blight and promptly remove infected leaves. Improve air circulation by proper spacing of plants."
        elif severity <= 40:
            return "Apply fungicides labeled for blight control as directed. Consider removing severely infected plants to prevent further spread."
        elif severity <= 70:
            return "Increase frequency of fungicide applications and remove severely infected plants. Avoid overhead watering to reduce moisture on foliage."
        else:
            return "Drastic measures may be necessary such as removing all infected plants, sterilizing tools and equipment, and implementing strict sanitation practices to prevent further spread."
    
    elif category == "Gray Leaf Spot":
        if severity <= 10:
            return "Monitor plants for early signs of grey leaf spot and remove infected leaves. Improve air circulation by proper spacing of plants."
        elif severity <= 40:
            return "Apply fungicides labeled for grey leaf spot control following manufacturer's instructions. Remove severely infected leaves."
        elif severity <= 70:
            return "Increase frequency of fungicide applications. Prune heavily infected parts of plants and dispose of them properly."
        else:
            return "Drastically reduce the spread by removing severely infected plants and implementing strict sanitation practices. Consider crop rotation for future seasons."
