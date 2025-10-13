from accounts.models import CustomUser
from django.db import models
from django.conf import settings
from django.utils import timezone

UNIT_CHOICE = [
    ("", "<no unit>"), ("g", "g"), ("kg", "kg"), ("ml", "ml"), ("l", "l"),
    ("tsp", "tsp"), ("tbsp", "tbsp"), ("cup", "cup"), ("piece", "piece"),
]

# Database 1
# Simple database for late optimization
class RecipeBasic(models.Model):
    recipe_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    servings = models.PositiveIntegerField(default=1)
    duration_minutes = models.PositiveIntegerField(default=0)
    is_public = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="data"
    )

    # JSON list of dicts: [{"order":0,"name":"potato","quantity":5,"unit":""}, ...]
    ingredients = models.JSONField(default=list)

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (#{self.pk})"
    
class RecipeString(models.Model):
    recipe = models.OneToOneField(
        RecipeBasic,
        on_delete=models.CASCADE,
        related_name="cook",
        primary_key=True,
    )
    raw_text = models.TextField(blank=True)
    saved_at = models.DateTimeField(default=timezone.now)


# How to look up the same recipe
# d = RecipeBasic.objects.get(pk=42) # or recipe_id=42
# r = RecipeString.objects.get(pk=42) # same id
