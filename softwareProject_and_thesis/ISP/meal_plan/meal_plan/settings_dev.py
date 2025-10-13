from .settings_common import *


# SECURITY WARNING: keep the secret key used in production secret
SECRET_KEY = 'django-insecure-$+2lu%rk%&(3irl1fh%n6h%8$9wpd@(bn770)nylz4h6q9-rpq'

# SECURITY WARNING: don't run which debug turned on in production!
DEBUG = True

# INSTALLED_APPS = [
#     'django.contrib.staticfiles',
#     'plans',
# ]

STATIC_URL = '/static/'

ALLOWED_HOSTS = []

# logging setting
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,

    # logger setting
    'loggers':{
        # the logger Django will use
        'django': {
            'handlers':['console'],
            'level':'INFO',
        },
        'recipe':{
            'handlers':['console'],
            'level': 'DEBUG',
        },
    },

    # handler setting
    'handlers':{
        'console':{
            'level':'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter':'dev'
        },
    },

    #formatter setting
    'formatters':{
        'dev':{
            'format': '\t'.join([
                '%(asctime)s'
                '[%(levelname)s]'
                '%(pathname)s(Line:%(lineno)d)',
                '%(message)s'
            ])
        },
    }
}


EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
DEFAULT_FROM_EMAIL = "noreply@localhost"
ACCOUNT_DEFAULT_HTTP_PROTOCOL = "http"

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')