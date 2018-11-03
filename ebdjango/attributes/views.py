from django.http import HttpResponse
from django.template import loader
from rest_framework.views import APIView
from rest_framework.response import Response
import os
import json

import tensorflow as tf
from tensorflow import keras

# from PIL import Image
import cv2
import numpy
from django.conf import settings
from attributes.align import resize
from attributes.utils import determine_attributes

def index(request):
    template = loader.get_template('attributes/index.html')

    context = {}
    return HttpResponse(template.render(context, request))

class TestView(APIView):
    def get(self, request, format=None):
        print (os.getcwd())
        print("asfdasdfa")
        p = os.path.join(os.getcwd(), 'model4.h5')
        # with open(p, 'r') as f:
        #   response = json.load(f)
        print(p)
        model = keras.models.load_model(p)
        print("loaded")

        model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        print("compiled")

        return Response({})

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']

        if myfile.content_type not in ('image/png', 'image/jpeg'):
            # some error handling
            return render(request, 'attributes/simple_upload.html')

        img = cv2.imdecode(numpy.fromstring(myfile.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)

        extension = myfile.name.split('.')[-1].lower()
        p = os.path.join(settings.BASE_DIR, 'static/ebdjango/input.png')
        with open(p, 'wb+') as dest:
            for chunk in myfile.chunks():
                dest.write(chunk)


        aligned_face = settings.ALIGNER.align_single_image(image=img, output_path='output.png')


        size = 30
        aligned_face = resize(image=aligned_face, new_width=size, is_square=True)
        aligned_face = aligned_face / 255
        # with open(p, 'r') as f:
        #   response = json.load(f)
        print("about to load")
        p = os.path.join(os.getcwd(), 'model4.h5')
        keras.backend.clear_session()
        model = keras.models.load_model(p)
        print("loaded")

        # model = settings.MODEL4

        model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        print("compiled")
        prediction = model.predict(numpy.expand_dims(aligned_face, axis=0), batch_size=1)
        attrs = determine_attributes(prediction[0])
        print(attrs)

        length = 0
        for key in ('sure', 'unsure'):
            for subkey in ('pos', 'neg'):
                if len(attrs[key][subkey]) > length:
                    length = len(attrs[key][subkey])
        print(length)
        for key in ('sure', 'unsure'):
            for subkey in ('pos', 'neg'):
                print(len(attrs[key][subkey]))
                print(length - len(attrs[key][subkey]))
                _length = length - len(attrs[key][subkey])
                attrs[key][subkey] += [""]*_length

        sure_pos = attrs['sure']['pos']
        print(sure_pos)
        unsure_pos = attrs['unsure']['pos']
        print(unsure_pos)
        sure_neg = attrs['sure']['neg']
        print(sure_neg)
        unsure_neg = attrs['unsure']['neg']
        print(unsure_neg)
        print()

        pretty_attrs = zip(sure_pos, unsure_pos, sure_neg, unsure_neg)

        print(dir(request))

        return render(request, 'attributes/simple_upload.html', context={'pretty_attrs': pretty_attrs})
    return render(request, 'attributes/simple_upload.html')

def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

