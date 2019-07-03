# import the necessary packages
from skimage.segmentation import clear_border
from imutils import contours
import cv2
import imutils
import numpy as np

charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))


def extract_digits_and_symbols(image, char_cnts, min_w=5, min_h=15):
    char_iter = char_cnts.__iter__()
    rois = []
    locs = []
    while True:
        try:
            c = next(char_iter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None
            if cW >= min_w and cH >= min_h:
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:
                parts = [c, next(char_iter), next(char_iter)]
                (s_xa, s_ya, s_xb, s_yb) = (np.inf, np.inf, -np.inf, -np.inf)
                for p in parts:
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    s_xa = min(s_xa, pX)
                    s_ya = min(s_ya, pY)
                    s_xb = max(s_xb, pX + pW)
                    s_yb = max(s_yb, pY + pH)
                roi = image[s_ya:s_yb, s_xa:s_xb]
                rois.append(roi)
                locs.append((s_xa, s_ya, s_xb, s_yb))
        except StopIteration:
            break
    return rois, locs


def find_ref_micr_contours(image):
    ref = imutils.resize(image, width=400)
    ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    ref_cnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_cnts = imutils.grab_contours(ref_cnts)
    ref_cnts = contours.sort_contours(ref_cnts, method="left-to-right")[0]
    return ref, ref_cnts


def find_ref_micr_data():
    image = cv2.imread('micr.png', 0)
    ref, ref_cnts = find_ref_micr_contours(image)
    ref_rois = extract_digits_and_symbols(ref, ref_cnts, min_w=10, min_h=20)[0]
    chars = {}
    for (name, roi) in zip(charNames, ref_rois):
        roi = cv2.resize(roi, (36, 36))
        chars[name] = roi
    return chars


def extract_blackhat(image):  # Here we want cheque image
    (h, w,) = image.shape[:2]
    delta = int(h - (h * 0.15))
    bottom = image[delta:h, 0:w]
    gray = np.copy(bottom)
    cv2.imwrite('bottom.jpg', gray)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    return blackhat, gray, delta


def find_group_contours(image):
    blackhat = extract_blackhat(image=image)[0]
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    (minVal, maxVal) = (np.min(grad_x), np.max(grad_x))
    grad_x = (255 * ((grad_x - minVal) / (maxVal - minVal)))
    grad_x = grad_x.astype("uint8")
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    group_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    group_cnts = imutils.grab_contours(group_cnts)
    return group_cnts


def group_locations(image):
    group_cnts = find_group_contours(image=image)
    group_locs = []
    for (i, c) in enumerate(group_cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 50 and h > 15:
            group_locs.append((x, y, w, h))
    group_locs = sorted(group_locs, key=lambda x: x[0])
    return group_locs


def extract_micr(image):
    blackhat, gray, delta = extract_blackhat(image=image)
    group_locs = group_locations(image=image)
    chars = find_ref_micr_data()
    output = []
    for (g_x, g_y, g_w, g_h) in group_locs:
        group_output = []
    group = gray[g_y - 2:g_y + g_h + 2, g_x - 2:g_x + g_w + 2]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    char_cnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_cnts = imutils.grab_contours(char_cnts)
    char_cnts = contours.sort_contours(char_cnts, method="left-to-right")[0]
    (rois, locs) = extract_digits_and_symbols(group, char_cnts)
    for roi in rois:
        scores = []
        roi = cv2.resize(roi, (36, 36))
        for charName in charNames:
            result = cv2.matchTemplate(roi, chars[charName], cv2.TM_CCORR)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        group_output.append(charNames[np.argmax(scores)])
    cv2.rectangle(image, (g_x - 10, g_y + delta - 10), (g_x + g_w + 10, g_y + g_y + delta), (0, 0, 255), 2)
    cv2.putText(image, "".join(group_output),
                (g_x - 10, g_y + delta - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
    output.append("".join(group_output))
    output = " ".join(output)
    return output, image
