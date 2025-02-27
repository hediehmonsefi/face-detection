import argparse


class Config:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Motion Detection")

        parser.add_argument("--image", type=str, required=True, help="Path to the image file")

        parser.add_argument("--input-size", type=int, nargs=2, default=(640, 640), help="Input size (width height)")
        parser.add_argument("--det-thresh", type=float, default=0.1, help="Threshold for detection")
        parser.add_argument("--ctx-id", type=float, default=0, help="Context ID (-1 for CPU, 0 for GPU, 1 for multi-GPU).")

        parser.add_argument("--show-keypoints", action="store_true", help="Visualize the key points")
        return parser.parse_args()