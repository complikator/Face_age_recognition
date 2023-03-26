import os
import cv2
import numpy

import pcn
from enum import Enum

class CallbackStatus(Enum):
    CONTINUE = 1
    STOP = 2

def detect_faces_in_image(source_image):
    """Detects faces from given image

    Parameters:
    source_image : str or numpy.ndarray
        Image or path to image

    Returns:
    faces : list of numpy.ndarray
        List of detected faces
    """

    if type(source_image) == str:
        source_image = cv2.imread(image)
    window_list = pcn.detect(img)
    faces = [f[0] for f in pcn.crop(img, window_list)]

    return faces

def detect_faces_from_webcam(callback):
    """Detects faces from webcam

    Parameters:
    callback : function
        This function is called with list of faces as argument
        and should return CallbackStatus.
        CallbackStatus.STOP results in loop break

        Example function for showing detected faces:
            `def f(faces):
                if len(faces) > 0:
                    faces = numpy.hstack(faces)
                    cv2.imshow("show", faces)
                    cv2.waitKey(1)
                return CallbackStatus.CONTINUE`

    """

    cam = cv2.VideoCapture(0)

    callback_result = CallbackStatus.CONTINUE

    while callback_result == CallbackStatus.CONTINUE:
        ret, img = cam.read()

        winlist = pcn.detect(img)
        faces = pcn.crop(img, winlist)
        faces = [f[0] for f in faces]

        callback_result = callback(faces)
