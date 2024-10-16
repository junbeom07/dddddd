from flask import Flask, render_template, send_from_directory, Response
import os
import cv2
import dlib
import numpy as np
from imutils import face_utils, resize
from datetime import datetime

app = Flask(__name__)

# 정적 파일 경로 설정
app.static_folder = 'assets'
# 얼굴 탐지기 및 랜드마크 예측기 초기화

orange_img = cv2.imread('orange.jpg')
orange_img = cv2.resize(orange_img, dsize=(512, 512))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shop')
def shop():
    return render_template('shop.html')

@app.route('/product-details')
def product_details():
    return render_template('product-details.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/vendor/<path:filename>')
def vendor_files(filename):
    return send_from_directory('vendor', filename)

@app.route('/cv')
def cv():
    return render_template('cv.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/md')
def md():
    return render_template('md.html')

@app.route('/dlibland') 
def dlibland():
    return render_template('dlibland.html')  

@app.route('/sticker') 
def sticker():
    return render_template('sticker.html')  

@app.route('/orange') 
def orange():
    return render_template('orange.html')  

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 얼굴 탐지
        faces = detector(gray)
        for face in faces:
            # 랜드마크 검출
            landmarks = predictor(gray, face)
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        # 프레임을 JPEG 형식으로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames2():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        faces = detector(img)
        result = orange_img.copy()

        if len(faces) > 0:
            face = faces[0]
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_img = img[y1:y2, x1:x2].copy()

            shape = predictor(img, face)
            shape = face_utils.shape_to_np(shape)

            for p in shape:
                cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

            # eyes
            le_x1 = shape[36, 0]
            le_y1 = shape[37, 1]
            le_x2 = shape[39, 0]
            le_y2 = shape[41, 1]
            le_margin = int((le_x2 - le_x1) * 0.18)

            re_x1 = shape[42, 0]
            re_y1 = shape[43, 1]
            re_x2 = shape[45, 0]
            re_y2 = shape[47, 1]
            re_margin = int((re_x2 - re_x1) * 0.18)

            left_eye_img = img[le_y1 - le_margin:le_y2 + le_margin, le_x1 - le_margin:le_x2 + le_margin].copy()
            right_eye_img = img[re_y1 - re_margin:re_y2 + re_margin, re_x1 - re_margin:re_x2 + re_margin].copy()

            left_eye_img = resize(left_eye_img, width=100)
            right_eye_img = resize(right_eye_img, width=100)

            result = cv2.seamlessClone(
                left_eye_img,
                result,
                np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
                (200, 220),
                cv2.MIXED_CLONE
            )

            result = cv2.seamlessClone(
                right_eye_img,
                result,
                np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
                (350, 220),
                cv2.MIXED_CLONE
            )

            # mouth
            mouth_x1 = shape[48, 0]
            mouth_y1 = shape[50, 1]
            mouth_x2 = shape[54, 0]
            mouth_y2 = shape[57, 1]
            mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

            mouth_img = img[mouth_y1 - mouth_margin:mouth_y2 + mouth_margin,
                        mouth_x1 - mouth_margin:mouth_x2 + mouth_margin].copy()

            mouth_img = resize(mouth_img, width=250)

            result = cv2.seamlessClone(
                mouth_img,
                result,
                np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
                (280, 320),
                cv2.MIXED_CLONE
            )

        # 결과 이미지를 JPEG 형식으로 인코딩
        ret, buffer = cv2.imencode('.jpg', result)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

class VideoCamera(object):
    def __init__(self):
        # 이미지 경로 지정
        self.sticker_path = 'sticker.png'  # 파일 이름 영어로 저장해야 함!
        self.img_hat = cv2.imread(self.sticker_path, cv2.IMREAD_UNCHANGED)

        # 모델 설정
        self.detector_hog = dlib.get_frontal_face_detector()
        self.model_path = 'shape_predictor_68_face_landmarks.dat'
        self.landmark_predictor = dlib.shape_predictor(self.model_path)

        # 스티커 크기 조절 변수
        self.scaling_factor_width = 1.0  # 기본 가로 크기
        self.scaling_factor_height = 1.0  # 기본 세로 크기

        # 원래 크기 저장
        self.original_scaling_factor_width = self.scaling_factor_width
        self.original_scaling_factor_height = self.scaling_factor_height

        # 스티커 위치 조정 변수
        self.sticker_offset_x = 0
        self.sticker_offset_y = 0

        # 원래 위치 저장
        self.original_sticker_offset_x = self.sticker_offset_x
        self.original_sticker_offset_y = self.sticker_offset_y

        # 웹캠 0번으로 설정
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None  # 프레임을 읽지 못한 경우

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_show = frame.copy()

        # 얼굴 검출
        dlib_rects = self.detector_hog(img_rgb, 1)

        # 여러 명 인식 가능하게 리스트를 만들어서 반복문
        list_landmarks = []
        for dlib_rect in dlib_rects:
            points = self.landmark_predictor(img_rgb, dlib_rect)
            list_points = list(map(lambda p: (p.x, p.y), points.parts()))
            list_landmarks.append(list_points)

        # 얼굴마다 스티커 합성
        for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
            # 코 위치 (30번 랜드마크) 기준으로 스티커 위치 계산
            x = landmark[30][0] + self.sticker_offset_x
            y = landmark[30][1] - dlib_rect.height() // 2 + self.sticker_offset_y
            w = int(dlib_rect.width() * self.scaling_factor_width)  # 가로 크기 조절
            h = int(dlib_rect.height() * self.scaling_factor_height)  # 세로 크기 조절

            # 스티커 이미지 크기 조절
            img_hat_resized = cv2.resize(self.img_hat, (w, h))

            refined_x = x - w // 2  # 모자 위치 조정
            refined_y = y - h

            # 경계 체크 및 조정
            if refined_x < 0:
                img_hat_resized = img_hat_resized[:, -refined_x:]
                refined_x = 0
            if refined_y < 0:
                img_hat_resized = img_hat_resized[-refined_y:, :]
                refined_y = 0

            # 스티커가 이미지 경계 밖으로 나가지 않도록 조정
            end_x = min(refined_x + img_hat_resized.shape[1], img_show.shape[1])
            end_y = min(refined_y + img_hat_resized.shape[0], img_show.shape[0])

            # 영역 조정
            img_hat_resized = img_hat_resized[:end_y-refined_y, :end_x-refined_x]
            refined_y = max(refined_y, 0)
            refined_x = max(refined_x, 0)

            # 합성할 영역 지정
            hat_area = img_show[refined_y:end_y, refined_x:end_x]

            # 알파 채널 처리하여 합성
            alpha_s = img_hat_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                hat_area[:, :, c] = (alpha_s * img_hat_resized[:, :, c] +
                                      alpha_l * hat_area[:, :, c])

            img_show[refined_y:end_y, refined_x:end_x] = hat_area

        # 결과 프레임 반환
        return img_show
    
def generate_frames3():
    while True:
        img_show = camera.get_frame()
        if img_show is None:
            break
        # JPEG 인코딩
        ret, buffer = cv2.imencode('.jpg', img_show)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 프레임 전송
    
@app.route('/video_feed3')
def video_feed3():
    return Response(generate_frames3(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

camera = VideoCamera()

if __name__ == '__main__':
    app.run(debug=True)
