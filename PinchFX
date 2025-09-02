import time
import cv2
import numpy as np
import mediapipe as mp

# Compatible with Windows – tested and running successfully. 

PINCH_THRESHOLD_PX = 30
PINCH_COOLDOWN_SEC = 0.30
TITLE = "EBI"
DETECT_CONF = 0.70
TRACK_CONF = 0.50

# load gear icon (png with alpha if possible)
gear = cv2.imread("C:\\Users\\ebi\\Desktop\\kk.png", cv2.IMREAD_UNCHANGED)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=DETECT_CONF,
    min_tracking_confidence=TRACK_CONF
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(TITLE, 1280, 720)

fx_idx = 0
last_pinch = 0.0
gear_angle = 0

fx_names = [
    "B&W","Invert","Thermal","Depth",
    "Gaussian Blur","Edges","Sketch","Cartoon",
    "Sepia","Sharpen","Pixelate","Hue Shift",
    "Emboss","CLAHE","Vignette","Chroma Shift",
    "Motion Blur","Saturation+","Brightness/Contrast+","Glass Distort"
]
FX_COUNT = len(fx_names)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    left_thumb = None
    left_pinky = None
    right_thumb = None
    right_pinky = None
    pinch = False

    if res.multi_hand_landmarks and res.multi_handedness:
        for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            lbl = handed.classification[0].label
            tt = lms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pk = lms.landmark[mp_hands.HandLandmark.PINKY_TIP]
            xt, yt = int(tt.x * W), int(tt.y * H)
            xp, yp = int(pk.x * W), int(pk.y * H)

            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            if np.hypot(xt - xp, yt - yp) < PINCH_THRESHOLD_PX:
                pinch = True

            if lbl == "Left":
                left_thumb = (xt, yt)
                left_pinky = (xp, yp)
            elif lbl == "Right":
                right_thumb = (xt, yt)
                right_pinky = (xp, yp)

    if left_thumb and left_pinky and right_thumb and right_pinky:
        poly = np.array([left_pinky, right_pinky, right_thumb, left_thumb], dtype=np.int32)
        filt = frame.copy()

        if fx_idx == 0:
            g = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
            filt = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        elif fx_idx == 1:
            filt = cv2.bitwise_not(filt)

        elif fx_idx == 2:
            g = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
            filt = cv2.applyColorMap(g, cv2.COLORMAP_JET)

        elif fx_idx == 3:
            g = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
            filt = cv2.applyColorMap(g, cv2.COLORMAP_BONE)

        elif fx_idx == 4:
            filt = cv2.GaussianBlur(filt, (21, 21), 0)

        elif fx_idx == 5:
            gray = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            filt = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif fx_idx == 6:
            gray = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            dodge = cv2.divide(gray, 255 - blur, scale=256)
            filt = cv2.cvtColor(dodge, cv2.COLOR_GRAY2BGR)

        elif fx_idx == 7:
            gray = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 2)
            color = cv2.bilateralFilter(filt, 9, 75, 75)
            edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            filt = cv2.bitwise_and(color, edges_col)

        elif fx_idx == 8:
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            filt = np.clip(filt @ kernel.T, 0, 255).astype(np.uint8)

        elif fx_idx == 9:
            k = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)
            filt = cv2.filter2D(filt, -1, k)

        elif fx_idx == 10:
            small = cv2.resize(filt, (max(1, W // 20), max(1, H // 20)), interpolation=cv2.INTER_LINEAR)
            filt = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)

        elif fx_idx == 11:
            hsv = cv2.cvtColor(filt, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h = (h.astype(np.int16) + 15) % 180
            hsv = cv2.merge([h.astype(np.uint8), s, v])
            filt = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif fx_idx == 12:
            k = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32)
            filt = cv2.filter2D(filt, -1, k)
            filt = cv2.add(filt, 128)

        elif fx_idx == 13:
            lab = cv2.cvtColor(filt, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge([l2, a, b])
            filt = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        elif fx_idx == 14:
            y, x = np.ogrid[:H, :W]
            cx, cy = W / 2, H / 2
            sigma = 0.6 * min(W, H)
            mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))
            mask = (mask * 255).astype(np.uint8)
            mask3 = cv2.merge([mask, mask, mask])
            dark = cv2.GaussianBlur(frame, (51, 51), 0)
            filt = cv2.addWeighted(frame, 0.8, dark, 0.2, 0)
            filt = ((filt.astype(np.float32) * (mask3 / 255.0)) + frame * (1 - (mask3 / 255.0))).astype(np.uint8)

        elif fx_idx == 15:
            b, g, r = cv2.split(filt)
            M = np.float32([[1, 0, 2], [0, 1, 0]])
            r = cv2.warpAffine(r, M, (W, H))
            M = np.float32([[1, 0, -2], [0, 1, 0]])
            b = cv2.warpAffine(b, M, (W, H))
            filt = cv2.merge([b, g, r])

        elif fx_idx == 16:
            ksz = 15
            kernel = np.zeros((ksz, ksz), np.float32)
            kernel[ksz // 2, :] = 1.0 / ksz
            filt = cv2.filter2D(filt, -1, kernel)

        elif fx_idx == 17:
            hsv = cv2.cvtColor(filt, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s.astype(np.int16) + 40, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h, s, v])
            filt = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif fx_idx == 18:
            alpha = 1.2
            beta = 20
            filt = np.clip(filt.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        elif fx_idx == 19:
            k = 5
            disp = 2
            m = np.indices((H, W)).transpose(1, 2, 0).astype(np.float32)
            m[:, :, 0] += ((np.sin(np.arange(H) / k)[:, None]) * disp).astype(np.float32)
            filt = cv2.remap(filt, m[:, :, 1], m[:, :, 0], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        mask3 = cv2.merge([mask, mask, mask]) // 255
        out = (filt * mask3 + frame * (1 - mask3)).astype(np.uint8)

        cv2.polylines(out, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)

        # gear rotation
        if gear is not None:
            gh, gw = gear.shape[:2]
            scale = 0.15
            new_w, new_h = int(W * scale), int(H * scale * gh / gw)
            gear_resized = cv2.resize(gear, (new_w, new_h))
            gear_angle += 15
            M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), gear_angle, 1.0)
            rotated = cv2.warpAffine(gear_resized, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            x_off, y_off = 20, 20
            if rotated.shape[2] == 4:
                alpha = rotated[:, :, 3] / 255.0
                for c in range(3):
                    out[y_off:y_off+new_h, x_off:x_off+new_w, c] = (alpha * rotated[:, :, c] +
                                                                   (1 - alpha) * out[y_off:y_off+new_h, x_off:x_off+new_w, c])
            else:
                out[y_off:y_off+new_h, x_off:x_off+new_w] = rotated

            cv2.putText(out, fx_names[fx_idx], (x_off + new_w + 10, y_off + new_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(TITLE, out)

        now = time.time()
        if pinch and (now - last_pinch) >= PINCH_COOLDOWN_SEC:
            fx_idx = (fx_idx + 1) % FX_COUNT
            last_pinch = now

    else:
        cv2.imshow(TITLE, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
