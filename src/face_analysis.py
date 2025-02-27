# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      :


from __future__ import division
import glob
import os.path as osp
import numpy as np
import onnxruntime
import cv2
from src.scrfd import SCRFD
from src.attributes.glasses import Glasses
from src.face import Face
import os


__all__ = ['FaceAnalysis', 'draw_on']



def get_model(onnx_file):
    """
    Loads the appropriate model based on the file name.

    :param onnx_file: Path to the ONNX model file.
    :return: Corresponding model object based on the file name.
             Returns None if no matching model is found.
    """
    file_name = os.path.basename(onnx_file)  # Extracts the file name
    if "det_10g" in file_name.lower():
        return SCRFD(model_file=onnx_file)  # Load Face Detection model
    elif "glasses" in file_name.lower():
        return Glasses(model_file=onnx_file)  # Load Glasses detection model
    # elif "beard" in file_name.lower():
    #    return Beard(model_file=onnx_file) # Load Beard detection model
    # elif "mask" in file_name.lower():
    #     return Mask(model_file=onnx_file)  # Load Mask detection model
    # elif "gender_age" in file_name.lower():
    #     return GenderAge(model_file=onnx_file)  # Load Gender detection and Age Estimation model
    else:
        return None


def draw_on(img, faces, show_keypoints):
    """
    Draws bounding boxes around detected faces and adds an information panel.
    :param img: Input image.
    :param faces: Detected face objects.
    :param show_keypoints: Boolean flag to visualize key points.
    :return: Image with drawn boxes and key points.
    """
    dimg = img.copy()
    height, width, _ = dimg.shape
    padding_width = 250  # Width of the black padding area
    padding_color = (0, 0, 0)  # Black background for text panel
    font_color = (255, 255, 255)  # White text

    # Create a new image with extra padding on the right
    padded_img = np.full((height, width + padding_width, 3), padding_color, dtype=np.uint8)
    padded_img[:, :width] = dimg  # Copy original image into the new image

    for i, face in enumerate(faces):
        box = face.bbox.astype(np.int64)
        cv2.rectangle(padded_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        if face.kps is not None and show_keypoints:
            kps = face.kps.astype(np.int64)
            for l in range(kps.shape[0]):
                color = (0, 255, 0) if l in [0, 3] else (0, 0, 255)
                cv2.circle(padded_img, (kps[l][0], kps[l][1]), 2, color, 2)

        # Prepare attributes text
        attributes = {
            "Glasses": "Yes" if face.glasses == 1 else "No" if face.glasses == 0 else "Unknown",
            "Beard": "Yes" if face.beard == 1 else "No" if face.beard == 0 else "Unknown",
            "Gender": "Male" if face.gender == 1 else "Female" if face.gender == 0 else "Unknown",
            "Age": str(face.age) if face.age is not None else "Unknown",
            "Mask": "Yes" if face.mask == 1 else "No" if face.mask == 0 else "Unknown",
        }

        # Add text to the black panel
        text_x = width + 10  # Start inside the black padding area
        text_y = 30  # Start from the top with padding
        line_height = 30  # Space between lines

        for key, value in attributes.items():
            text = f"{key}: {value}"
            cv2.putText(padded_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, font_color, 2, cv2.LINE_AA)
            text_y += line_height  # Move down for the next line

    return padded_img


class FaceAnalysis:
    def __init__(self, root='models/', allowed_modules=None, **kwargs):
        """
        Initializes the face analysis models.

        :param root: Path to the directory containing model files (default: 'models/').
        :param allowed_modules: List of specific models to load (e.g., ['detection', 'glasses', 'beard']).
                                If None, all available models are loaded.
        :param kwargs: Additional keyword arguments for model configuration.
        """
        self.det_size = None
        self.det_thresh = None
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = root
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)

        for onnx_file in onnx_files:
            model = get_model(onnx_file)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (
                    allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape,
                      model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)


    def get(self, img, max_num=0, det_metric='default'):
        """
        Detects faces in an image and applies attribute analysis.

        :param img: Input image as a NumPy array (BGR format).
        :param max_num: Maximum number of faces to return. If 0, returns all detected faces.
        :param det_metric: Metric for sorting detected faces ('default' sorts by confidence,
                           'max' prioritizes largest faces).
        :return: List of Face objects, each containing bounding box, key points, detection score,
                 and additional attributes like glasses, beard, gender, age, and mask (if available).
        """
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric=det_metric)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for task_name, model in self.models.items():
                if task_name == 'detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

