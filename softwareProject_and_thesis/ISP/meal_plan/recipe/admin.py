from django.contrib import admin

from .models import RecipeBasic, RecipeString

admin.site.register(RecipeBasic)
admin.site.register(RecipeString)
