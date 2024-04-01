# from django.urls import path
# from . import views

# urlpatterns = [
#     path('signup/', views.signup, name='signup'),
#     path('login/', views.login, name='login'),
#     path('home/', views.home, name='home'),
# ]


from django.urls import path
from .views import home, user_login, user_signup, image_upload, image_upload_success, logout_view, user_signup_login




urlpatterns = [
    path('', home, name='home'),
    path('login/', user_login, name='login'),
    path('signup/', user_signup, name='signup'),
    path('SignupLogin/', user_signup_login, name='signup_login'),
    path('image/upload/', image_upload, name='image_upload'),
    path('image/upload/success/', image_upload_success, name='image_upload_success'),
    #path('image/upload/success/<str:suggestion>/', image_upload_success, name='image_upload_success'),
    path('logout/', logout_view, name='logout'),
    # Other URL patterns
]
