
from imutils import face_utils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
non_ENM_landmarks_list = list(range(0, 17)) # Excluded landmarks

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
            # print("rec__"+str(len(rec)), list(rec))
            found_face = (False if ((len(rec) == 0) or (found_face == False)) else True)
        except:
            found_face = False

        if found_face:
            # print("Face", found_face)
            for (i, re) in enumerate(rec):
                sh = face_utils.shape_to_np(predictor(sub_face, re))

                lands = []
                for j, xy in enumerate(sh):
                    (xx, yy) = xy
                    if not ((j in non_ENM_landmarks_list) and (non_ENM_landmarks)):
                        lands.append((xx, yy))
                        cv2.circle(sub_face, (xx, yy), 0, (0, 255, 0), -1)
                        # cv2.putText(sub_face, str(j), (xx, yy), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 2)
                face_landmarks.append(lands)
                if showImg:
                    cv2.imshow(image_dir, sub_face)
                    cv2.waitKey(0)
            faces.append(sub_face)
            face_landmarks_all.append(face_landmarks)
            # print(sh)
        else:
            if showImg:
                cv2.imshow(image_dir, image)
                cv2.waitKey(0)
                print("NOT FOUND")

    return faces, face_landmarks_all, found_face

# print(getFacialLandmarks('my_image.png', showImg=True))
