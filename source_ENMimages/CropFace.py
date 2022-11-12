
from imutils import face_utils
import dlib
import cv2
from PIL import Image
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
non_ENM_landmarks_list = list(range(0, 17)) # Excluded landmarks
ENM_landmarks_list = np.array(list(range(17, 64)))
crop_pixel_sqlen_list = {}

def pixel_radius_assigner(dict, pixel_len, start, end):
    for i in range(start, end):
        dict[i] = pixel_len

    return dict

crop_pixel_sqlen_list = pixel_radius_assigner(crop_pixel_sqlen_list, 5, start=17, end=28)   # eye brow
crop_pixel_sqlen_list = pixel_radius_assigner(crop_pixel_sqlen_list, 12, start=28, end=36)  # nose
crop_pixel_sqlen_list = pixel_radius_assigner(crop_pixel_sqlen_list, 9, start=36, end=48)   # eyes
crop_pixel_sqlen_list = pixel_radius_assigner(crop_pixel_sqlen_list, 9, start=48, end=68)  # mouth


def pixel_radius_highlighter(highlight_image_array, x_y, length, highlight_num=0):
    highlight_image_array = np.array(highlight_image_array)
    x = x_y[0]
    y = x_y[1]
    for i in range(max(0, y - length), min(y + length, highlight_image_array.shape[0])):
        for j in range(max(0, x - length), min(x + length, highlight_image_array.shape[1])):
             highlight_image_array[i, j] = highlight_num

    return highlight_image_array


def remove_non_highlighted_pixels(image_array, highlighted_image_array, highlight_num=0):
    image_array = np.array(image_array)
    for i in range(0, image_array.shape[0]):
        for j in range(0, image_array.shape[1]):
            if highlighted_image_array[i, j] != highlight_num:
                image_array[i, j] = 0

    return image_array


def getFacialLandmarks(image_dir='', img=None, showImg=True, non_ENM_landmarks=True):

    if len(image_dir) >= 1:
        image = cv2.imread(image_dir, flags=cv2.IMREAD_GRAYSCALE)
    else:
        # img = cv2.resize(img, (120, 90))
        image = img
    image = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT)

    gray = image

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    found_face = (False if len(rects) == 0 else True)

    face_landmarks_all = []
    faces = []
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        face_landmarks = []

        # print("rect", rect)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        x -= 10 if w < 100 else int(w / 8)
        y -= 10 if w < 100 else int(w / 8)
        w += 20 if w < 100 else int(w / 4)
        h += 20 if w < 100 else int(w / 4)

        sub_face_ = image[y:y + h, x:x + w]

        try:
            sub_face = cv2.resize(sub_face_, (100, 100))
            rec = detector(sub_face, 1)
            # rec = rects
            # print("rec__"+str(len(rec)), list(rec))
            found_face = (False if ((len(rec) == 0) or (found_face == False)) else True)
        except:
            found_face = False

        if found_face:
            # print("Face", found_face)
            for (i, re) in enumerate(rec):
                sh = face_utils.shape_to_np(predictor(sub_face, re))

                lands = []
                highlighted_image_array = np.array(sub_face)
                for j, xy in enumerate(sh):
                    (xx, yy) = xy
                    if not ((j in non_ENM_landmarks_list) and (non_ENM_landmarks)):
                        lands.append((xx, yy))
                        # cv2.circle(sub_face, (xx, yy), 5, (0, 255, 0), 1)
                        # cv2.putText(sub_face, str(j), (xx, yy), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 0)

                        highlighted_image_array = pixel_radius_highlighter(highlighted_image_array, xy, length=crop_pixel_sqlen_list[j])
                ENM_cropped_image_array = remove_non_highlighted_pixels(sub_face, highlighted_image_array)
                face_landmarks.append(lands)
                if showImg:
                    # cv2.imshow(image_dir, sub_face)
                    Image.fromarray(highlighted_image_array).show()
                    Image.fromarray(ENM_cropped_image_array).show()
                    # Image.fromarray(sub_face).show()
                    cv2.waitKey(0)
            faces.append(ENM_cropped_image_array)
            # Image.fromarray(sub_face).show()
            # Image.fromarray(ENM_cropped_image_array).show()
            # faces.append(sub_face)
            face_landmarks_all.append(face_landmarks)
            # print(sh)
        else:
            if showImg:
                cv2.imshow(image_dir, image)
                cv2.waitKey(0)
                print("NOT FOUND")

    return faces, face_landmarks_all, found_face

# print(getFacialLandmarks('../input_files/obama.jpg', showImg=True))
