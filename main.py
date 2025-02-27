import cv2
from src.face_analysis import FaceAnalysis, draw_on
from src.config import Config


def main():
    args = Config.parse_args()

    face_analyzer = FaceAnalysis()
    face_analyzer.prepare(-1)

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError("Image not found!")

    # Detect faces and Process each face and attributes
    faces = face_analyzer.get(image)
    output_image = draw_on(image, faces, show_keypoints=args.show_keypoints)
    cv2.imshow("result", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()