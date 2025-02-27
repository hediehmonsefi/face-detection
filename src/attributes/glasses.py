from src.attributes.attribute import Attribute
import numpy as np
import cv2
import onnx
import onnxruntime

class Glasses(Attribute):
    def __init__(self, model_file, session=None):
        """
        Initializes the glasses detection model.

        :param model_file: Path to the ONNX model.
        :param session: Optional ONNX session.
        """
        super().__init__(model_file, session)
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        model = onnx.load(self.model_file)
        self.input_mean = 127.5
        self.input_std = 128.0
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        # input_name = input_cfg.name
        self.input_size = tuple(input_shape[1:3][::-1])
        self.input_shape = input_shape
        # outputs = self.session.get_outputs()
        # output_names = []
        # for out in outputs:
        #     output_names.append(out.name)
        # self.input_name = input_name
        # self.output_names = output_names
        # assert len(self.output_names) == 1
        # output_shape = outputs[0].shape
        # if output_shape[1]==1:
        #     self.taskname = 'glasses'
        # else:
        #     self.taskname = 'attribute_%d'%output_shape[1]
        self.taskname = 'glasses'


    def prepare(self, ctx_id, **kwargs):
        """
        Prepares the session for CPU or GPU execution.

        :param ctx_id: Context ID (negative for CPU, 0 for GPU).
        """
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])


    def get(self, img, face):
        """
        Detects if the face in the image has glasses.

        :param img: Input image.
        :param face: Face object with bounding box.
        :return: 1 if glasses are detected, else 0.
        """
        bbox = face.bbox.astype(int)
        face_crop = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        face_crop = cv2.resize(face_crop, [160, 160])
        face_crop = np.expand_dims(face_crop, axis=0).astype(np.float32)

        pred = self.session.run(self.output_names, {self.input_name: face_crop})
        pred = np.squeeze(pred)

        glasses = int(np.round(pred))
        face['glasses'] = glasses
        return glasses