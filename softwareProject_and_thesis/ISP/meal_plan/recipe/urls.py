from django.urls import path
from .import views

app_name = 'recipe'
urlpatterns = [
    path('', views.IndexView.as_view(), name="index"),
    path('recipe-list/', views.RecipeListView.as_view(), name="recipe_list"),
    path('recipe-detail/<int:pk>/', views.RecipeDetailView.as_view(), name="recipe_detail"),
    path('recipe-create/', views.RecipeCreateView.as_view(), name="recipe_create"),
    path('recipe-update/<int:pk>/', views.RecipeUpdateView.as_view(), name="recipe_update"),
    path('recipe-delete/<int:pk>/', views.RecipeDeleteView.as_view(), name="recipe_delete"),

]
