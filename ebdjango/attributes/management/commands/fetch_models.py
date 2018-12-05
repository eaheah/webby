from django.core.management.base import BaseCommand, CommandError
from attributes.get_models import list_objects, download_file
from django.conf import settings

import json
import os

from attributes.utils import md

class Command(BaseCommand):

    def handle(self, *args, **options):

    	directory = os.path.join(settings.BASE_DIR, 'agegender_model')
    	md(directory)
    	bucket = "django-attributes-bucket"
    	# l = list_objects(bucket)
    	keys = [l['Key'] for l in list_objects(bucket)['Contents']]
    	for key in keys:
    		download_file(bucket, key, '{}/{}'.format(directory, key))

    	directory = os.path.join(settings.BASE_DIR, 'emotion_model')
    	md(directory)

    	bucket = "django-emotion-bucket"

    	keys = [l['Key'] for l in list_objects(bucket)['Contents']]
    	for key in keys:
    		download_file(bucket, key, '{}/{}'.format(directory, key))