import cv2
import traceback

def facechop(image, ind=0, store=True, imgShow=True):
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, borderType=cv2.BORDER_CONSTANT)
    # cv2.imshow(image, img)
    # cv2.waitKey(10000)

    # minisize = (img.shape[0], img.shape[1])
    # img = cv2.resize(img, (640, 480))
    # cv2.imshow(image, img)
    # cv2.waitKey(10000)

    faces = cascade.detectMultiScale(img)
    # p1 = (292 - 143, 501 - 219)  # Landmark of a
    # (pointX - imgX, pointY - imgY)

    foundFace = len(faces) > 0
    if foundFace:
        # for f in faces:
        x, y, w, h = [v for v in faces[0]]
        print((w, h))
    else:
        foundFace = False
    try:
        if foundFace:
            if ((w >= 120 or h >= 120) and foundFace == True):

                # w = (w if w >= h else h)
                # h = w

                x -= 75
                y -= 75
                w += 150
                h += 150
                print("w,h", (y + h, x + h))

                print("Image size", (img.shape[1], img.shape[0]))
                print("Face x, y, w, h", (x, y, w, h))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
                # cv2.rectangle(img, p1, p1, (0, 255, 0))
                # x,y=p1
                # w,h=p1

                sub_face = img[y:y + h, x:x + w]
                sub_face = cv2.resize(sub_face, (500, 500))

                if imgShow:
                    cv2.imshow(image, sub_face)
                    cv2.waitKey(1000)

                print(sub_face.shape)
                face_file_name = "faces/face_" + str(ind) + ".jpg"
                if store:
                    cv2.imwrite(face_file_name, sub_face)
                if imgShow:
                    cv2.imshow(image, img)
                    cv2.waitKey(100000)
            else:
                foundFace = False
                sub_face = []
                x = y = w = h = None
                print("NF")
        else:
            foundFace = False
            sub_face = []
            x = y = w = h = None
            print("NF")
    except:
        traceback.print_exc()
        foundFace = False
        sub_face = []
        x = y = w = h = None
        print("NF")
    # x and y are the coordinates of the top left of the image(imgX, imgY)
    return sub_face, (x, y, w, h), foundFace

