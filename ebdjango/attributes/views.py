import os
import json
import cv2
import numpy

import tensorflow as tf
from tensorflow import keras

from django.conf import settings
from django.http import HttpResponseRedirect
from django.views.generic.base import TemplateView
from django.shortcuts import render

from attributes.align import resize
from attributes.utils import determine_attributes

from attributes import jiang_eval
import time
t = time.time

from PIL import Image
import io



def create_model(input_shape=(48,48,3), optimizer=tf.train.AdamOptimizer, loss='binary_crossentropy', metrics=['accuracy']):
    model = keras.Sequential([
        keras.layers.Conv2D(20, (4,4), activation=tf.nn.relu, input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Dropout(0.25),
        keras.layers.Conv2D(40, (3,3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Dropout(0.25),
        keras.layers.Conv2D(60, (3,3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Dropout(0.25),
        keras.layers.Conv2D(80, (2,2), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(40, activation=tf.nn.sigmoid)
      ])
    model.compile(optimizer=optimizer(),
                  loss=loss,
                  metrics=metrics)
    return model

def _agegender_model(myfile):

    img = cv2.imdecode(numpy.fromstring(myfile.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    p = os.path.join(settings.BASE_DIR, 'static/attributes/input.png')
    with open(p, 'wb+') as dest:
        for chunk in myfile.chunks():
            dest.write(chunk)

    print(img.shape)
    print("AAA")
    s = t()
    aligned_image, image, rect_nums, XY = jiang_eval.load_image(img, os.path.join(settings.BASE_DIR, 'attributes/shape_predictor_68_face_landmarks.dat'))
    print("load: {}".format(t() - s))
    s = t()
    ages, genders = jiang_eval.eval(aligned_image, os.path.join(settings.BASE_DIR, 'agegender_model'))
    print("eval: {}".format(t() - s))
    print(ages, genders)

    result = {'age': int(round(ages[0])), 'gender': 'Female' if genders[0] == 0 else 'Male'}

    return result

class AgeGenderView(TemplateView):
    template_name = 'agegender.html'
    def get(self, request, *args, **kwargs):
        '''Redirect to login if not authenticated or process GET request.'''
        #if not request.user.is_authenticated():
        #    return HttpResponseRedirect(reverse("login"))
        return super(AgeGenderView, self).get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        if request.FILES['attributes']:
            myfile = request.FILES['attributes']
            if myfile.content_type not in ('image/png', 'image/jpeg'):
                return HttpResponseRedirect(reverse("agegender"))

            guess = _agegender_model(myfile)
            context = self.get_context_data()
            context['guess'] = guess

            return super(AgeGenderView, self).render_to_response(context)

        else:
            return HttpResponseRedirect(reverse("agegender"))

    def get_context_data(self, **kwargs):
        context = super(AgeGenderView, self).get_context_data(**kwargs)

        return context

def _attributes_model(myfile):


    img = cv2.imdecode(numpy.fromstring(myfile.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)

    extension = myfile.name.split('.')[-1].lower()
    p = os.path.join(settings.BASE_DIR, 'static/attributes/input.png')
    # with open(p, 'wb+') as dest:
    #     for chunk in myfile.chunks():
    #         dest.write(chunk)

    _image = Image.open(myfile)
    # _image.thumbnail((10, 10))
    _image.save(p, 'PNG')


    aligned_face = settings.ALIGNER.align_single_image(image=img, output_path='output.png')


    size = 48
    aligned_face = resize(image=aligned_face, new_width=size, is_square=True)
    aligned_face = aligned_face / 255
    # with open(p, 'r') as f:
    #   response = json.load(f)
    print("about to load")
    p = os.path.join(os.getcwd(), 'model63/cp-0038.ckpt')
    keras.backend.clear_session()
    model = create_model()
    model.load_weights(p)
    # model = keras.models.load_model(p)
    print("loaded")

    # model = settings.MODEL4

    model.compile(optimizer=tf.train.AdamOptimizer(),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    print("compiled")
    prediction = model.predict(numpy.expand_dims(aligned_face, axis=0), batch_size=1)
    attrs, results_as_string = determine_attributes(prediction[0])
    # print(attrs)

    length = 0
    for key in ('sure', 'unsure'):
        for subkey in ('pos', 'neg'):
            if len(attrs[key][subkey]) > length:
                length = len(attrs[key][subkey])
    # print(length)
    for key in ('sure', 'unsure'):
        for subkey in ('pos', 'neg'):
            # print(len(attrs[key][subkey]))
            # print(length - len(attrs[key][subkey]))
            _length = length - len(attrs[key][subkey])
            attrs[key][subkey] += [""]*_length

    sure_pos = attrs['sure']['pos']
    # print(sure_pos)
    unsure_pos = attrs['unsure']['pos']
    # print(unsure_pos)
    sure_neg = attrs['sure']['neg']
    # print(sure_neg)
    unsure_neg = attrs['unsure']['neg']
    # print(unsure_neg)
    # print()

    pretty_attrs = zip(sure_pos, unsure_pos, sure_neg, unsure_neg)


    return pretty_attrs, results_as_string

class AttributesView(TemplateView):
    template_name = 'attributes.html'

    def get(self, request, *args, **kwargs):
        '''Redirect to login if not authenticated or process GET request.'''
        #if not request.user.is_authenticated():
        #    return HttpResponseRedirect(reverse("login"))
        return super(AttributesView, self).get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        if request.FILES['attributes']:
            myfile = request.FILES['attributes']
            if myfile.content_type not in ('image/png', 'image/jpeg'):
                return HttpResponseRedirect(reverse("attributes"))

            pretty_attrs, results_as_string = _attributes_model(myfile)
            context = self.get_context_data()
            context['pretty_attrs'] = pretty_attrs
            context['results_as_string'] = results_as_string

            return super(AttributesView, self).render_to_response(context)

        else:
            return HttpResponseRedirect(reverse("attributes"))

    def get_context_data(self, **kwargs):
        context = super(AttributesView, self).get_context_data(**kwargs)

        return context

def _emotion_model(myfile):


    img = cv2.imdecode(numpy.fromstring(myfile.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)


    p = os.path.join(settings.BASE_DIR, 'static/attributes/input.png')
    with open(p, 'wb+') as dest:
        for chunk in myfile.chunks():
            dest.write(chunk)


    aligned_face = settings.ALIGNER.align_single_image(image=img, output_path='output.png')

    size = 48
    aligned_face = resize(image=aligned_face, new_width=size, is_square=True)
    print(aligned_face.shape)
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    print(aligned_face.shape)
    # aligned_face = aligned_face / 255 #?
    aligned_face = numpy.reshape(aligned_face, [-1, 48, 48, 1])
    print(aligned_face.shape)

    p = os.path.join(os.getcwd(), 'emotion_model/my_version.h5')
    keras.backend.clear_session()
    model = keras.models.load_model(p)

    model.compile(optimizer=tf.train.AdamOptimizer(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    print("compiled")
    # print(numpy.expand_dims(aligned_face, axis=0).shape)
    prediction = model.predict(aligned_face, batch_size=1)

    print("loaded")
    # print(prediction)
    # print(numpy.argmax(prediction[0]))
    # print(len(prediction[0]))

    emos = ["Angry", 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    emotion = emos[numpy.argmax(prediction[0])]
    # print(emotion)

    prediction = list(prediction[0])

    print(prediction)
    print(emos)

    ordered_emotions = []

    one = numpy.argmax(prediction)
    prediction.pop(one)
    ordered_emotions.append(emos.pop(one))
    print(prediction)
    print(emos)
    print(ordered_emotions)

    one = numpy.argmax(prediction)
    prediction.pop(one)
    ordered_emotions.append(emos.pop(one))
    print(prediction)
    print(emos)
    print(ordered_emotions)

    one = numpy.argmax(prediction)
    prediction.pop(one)
    ordered_emotions.append(emos.pop(one))
    print(prediction)
    print(emos)
    print(ordered_emotions)

    one = numpy.argmax(prediction)
    prediction.pop(one)
    ordered_emotions.append(emos.pop(one))
    print(prediction)
    print(emos)
    print(ordered_emotions)

    one = numpy.argmax(prediction)
    prediction.pop(one)
    ordered_emotions.append(emos.pop(one))
    print(prediction)
    print(emos)
    print(ordered_emotions)

    one = numpy.argmax(prediction)
    prediction.pop(one)
    ordered_emotions.append(emos.pop(one))
    print(prediction)
    print(emos)
    print(ordered_emotions)



    return {i: val for i, val in enumerate(ordered_emotions)}

class EmotionView(TemplateView):
    template_name = 'emotion.html'

    def get(self, request, *args, **kwargs):
        '''Redirect to login if not authenticated or process GET request.'''
        #if not request.user.is_authenticated():
        #    return HttpResponseRedirect(reverse("login"))
        return super(EmotionView, self).get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        if request.FILES['attributes']:
            myfile = request.FILES['attributes']
            if myfile.content_type not in ('image/png', 'image/jpeg'):
                return HttpResponseRedirect(reverse("emotion"))

            guess = _emotion_model(myfile)
            context = self.get_context_data()
            context['guess'] = guess
            return super(EmotionView, self).render_to_response(context)

        else:
            return HttpResponseRedirect(reverse("emotion"))

    def get_context_data(self, **kwargs):
        context = super(EmotionView, self).get_context_data(**kwargs)

        return context



