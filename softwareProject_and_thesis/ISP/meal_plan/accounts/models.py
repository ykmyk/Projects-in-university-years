from django.contrib.auth.models import AbstractUser


class CustomUser(AbstractUser):
    """extended user model"""

    class Meta:
        verbose_name_plural = 'CustomUser'