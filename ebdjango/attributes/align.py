import numpy as np
import cv2
import dlib
import os
from matplotlib import pyplot as plt
import time

t = time.time


def resize(image, new_width, is_square=False, interpolation=cv2.INTER_AREA):
    if is_square:
        return cv2.resize(image, (new_width, new_width), interpolation=interpolation)
    else:
        img_height, img_width = image.shape[:2]
        new_height = int(new_width / float(img_width) * img_height)
        return cv2.resize(image, (new_width, new_height))

class NoFaceException(Exception):
    pass

class Aligner:
    def __init__(self, face_detector, output_path, predictor_path, save=False):
        self.face_detector = face_detector
        self.save = save
        self.output_path = output_path
        s = t()
        self.predictor = dlib.shape_predictor(predictor_path)
        print("shape_predictor init took: {}".format(t() - s))
        self.output_left_eye = (0.35, 0.35)
        # self.output_width = 256
        # self.output_height = 256
        self.bbox_shape = 95

    def get_bbox(self, dlib_bbox):
        left_x = dlib_bbox.left()
        top_y = dlib_bbox.top()
        width = dlib_bbox.width()
        height = dlib_bbox.height()

        return left_x, top_y, width, height

    def mod_path(self, image_path):
        return image_path.replace('jpg', 'png')

    def save_figure(self, original_image, aligned_image, box_args, image_path):
        image_path = self.mod_path(image_path)
        # print(original_image.dtype)
        filename = os.path.join(self.output_path, image_path)
        print(filename)
        # rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # for center in (self.left_center, self.right_center):
        #     cv2.circle(rgb_image, center, 5, color=(0,0,255), thickness=-1)
        # original_image = self.bound(rgb_image, box_args)

        gray_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        # aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)

        boxes = self.face_detector(gray_image, 1)
        if len(boxes) == 1:
            box_args = self.get_bbox(boxes[0])

            aligned_image = self.bound(aligned_image, box_args)
            print(aligned_image.shape)

            cv2.imwrite(filename, aligned_image)

            aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
            if aligned_image.shape[-1] == 4:
                aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_RGBA2RGB)
            print(aligned_image.shape)
            return aligned_image


        else:
            raise NoFaceException('No Face / Multiple Faces Detected')

    def get_coordinates(self, prediction):
        coordinates = np.zeros((prediction.num_parts, 2), dtype="int")
        for i in range(0, prediction.num_parts):
            coordinates[i] = (prediction.part(i).x, prediction.part(i).y)
        return coordinates

    def bound(self, image, args):
        # print("bound args")
        # print(args)
        return resize(image[args[1]:args[1] + args[3], args[0]:args[0] + args[2]], self.bbox_shape, is_square=True)

    def get_rotation(self, coordinates):
        left_eye = coordinates[42:48]
        right_eye = coordinates[36:42]

        left_center = left_eye.mean(axis=0).astype("int")
        right_center = right_eye.mean(axis=0).astype('int')
        # print("left center, right center")
        # print(left_center, right_center)
        self.left_center = tuple(left_center)
        self.right_center = tuple(right_center)

        x = right_center[0] - left_center[0]
        y = right_center[1] - left_center[1]

        angle = np.degrees(np.arctan2(y, x)) - 180

        new_right_x = 1.0 - self.output_left_eye[0]

        distance = np.sqrt((x**2) + (y**2))
        new_distance = new_right_x - self.output_left_eye[0]
        new_distance = new_distance * self.output_width
        scale = new_distance / distance

        center = (left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2
        return center, angle, scale

    def align(self, img, img_gray, bbox, image_path):
        box_args = self.get_bbox(bbox)
        predition = self.predictor(img_gray, bbox)
        coordinates = self.get_coordinates(predition)


        center, angle, scale = self.get_rotation(coordinates)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # print(center, angle, scale)

        tX = self.output_width * 0.5
        tY = self.output_height * self.output_left_eye[1]

        rotation_matrix[0, 2] += (tX - center[0])
        rotation_matrix[1,2] += (tY - center[1])

        aligned_face = cv2.warpAffine(img, rotation_matrix, (self.output_width, self.output_height), flags=cv2.INTER_CUBIC)

        return self.save_figure(img, aligned_face, box_args, image_path)

    def get_largest_bbox(self, boxes):
        area = 0
        box = None
        for box in boxes:
            box_area = box.width() * box.height()
            box = box if box_area > area else box
            area = box_area if box_area > area else area
        return box

    def generate(self, path):
        for image_path in os.listdir(path):

            image = cv2.imread(os.path.join(path, image_path))
            # image = resize(image, new_width=800)
            self.output_width, self.output_height = image.shape[:2]

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            boxes = self.face_detector(gray_image, 1)# number of times image is upsampled (more means more faces, but may not need much here)

            if boxes:
                if not len(boxes) == 1:
                    box = self.get_largest(boxes)
                else:
                    box = boxes[0]
                yield face_aligner.align(image, gray_image, box, image_path)
            else:
                yield None

    def align_single_image(self, image=None, path=None, output_path=None):
        assert not (image is None and path is None)
        assert output_path is not None
        if image is not None:
            assert path is None

        elif path is not None:
            assert image is None
            image = cv2.imread(path)

        self.output_width, self.output_height = image.shape[:2]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boxes = self.face_detector(gray_image, 1)

        if boxes:
            if not len(boxes) == 1:
                box = self.get_largest(boxes)
            else:
                box = boxes[0]

            return self.align(image, gray_image, box, output_path)

        else:
            raise NoFaceException('No Face Detected')



if __name__ == '__main__':

    s = t()
    face_detector = dlib.get_frontal_face_detector()
    print("getting dlib face face_detector took: {}".format(t() - s))

    s = t()
    face_aligner = Aligner(face_detector=face_detector, output_path='/vagrant/ebdjango/aligned/', predictor_path='shape_predictor_68_face_landmarks.dat', save=True)
    print("class instantiation took: {}".format(t() - s))


    s = t()
    aligned_image = face_aligner.align_single_image(path='/vagrant/ebdjango/000001.jpg', output_path='000001.jpg')
    print("aligning image took: {}".format(t() - s))




