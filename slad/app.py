from flask import Flask, render_template, request, send_from_directory, Response
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
selected_image_path = 'orange.jpg'

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


@app.route('/md')
def md():
    return render_template('md.html')

@app.route('/dlibland') 
def dlibland():
    return render_template('dlibland.html')  

@app.route('/face swap')
def face_swap():
    return render_template('face swap.html')

@app.route('/sticker') 
def sticker():
    # 업로드된 스티커 목록을 가져오기
    sticker_files = [f for f in os.listdir('stickers') if f.startswith('sticker') and f.endswith('.png')]  # stickers 폴더에서 파일 목록 가져오기
    return render_template('sticker.html', sticker_files=sticker_files)  # 스티커 파일 목록을 템플릿에 전달

@app.route('/orange') 
def orange():
    return render_template('orange.html')  

@app.route('/select_image')
def select_image():
    return render_template('select_image.html')

@app.route('/set_image', methods=['POST'])
def set_image():
    global selected_image_path
    selected_image_path = request.form['image']  # 선택된 이미지 경로 저장
    image_name = selected_image_path.split('.')[0]  # 이미지 이름 추출 (확장자 제외)
    return render_template('orange.html', image_name=image_name)  # 이미지 이름을 템플릿에 전달

@app.route('/deep') 
def deep():
    return render_template('deep.html')  


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
    orange_img = cv2.imread(selected_image_path)  # 사용자가 선택한 이미지 사용
    orange_img = cv2.resize(orange_img, dsize=(512, 512))

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

            # 눈의 좌표
            le_x1 = shape[36, 0]
            le_y1 = shape[37, 1]
            le_x2 = shape[39, 0]
            le_y2 = shape[41, 1]
            le_margin = int((le_x2 - le_x1) * 0.2)  # 마진 증가

            re_x1 = shape[42, 0]
            re_y1 = shape[43, 1]
            re_x2 = shape[45, 0]
            re_y2 = shape[47, 1]
            re_margin = int((re_x2 - re_x1) * 0.2)  # 마진 증가

            # 눈 이미지 추출
            left_eye_img = img[le_y1 - le_margin:le_y2 + le_margin, le_x1 - le_margin:le_x2 + le_margin].copy()
            right_eye_img = img[re_y1 - re_margin:re_y2 + re_margin, re_x1 - re_margin:re_x2 + re_margin].copy()

            # 눈 이미지 크기 조정
            left_eye_img = resize(left_eye_img, width=120)  # 크기 증가
            right_eye_img = resize(right_eye_img, width=120)  # 크기 증가

            # 눈 합성
            result = cv2.seamlessClone(
                left_eye_img,
                result,
                np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
                (180, 220),  # 위치를 왼쪽으로 20만큼 이동
                cv2.MIXED_CLONE
            )

            result = cv2.seamlessClone(
                right_eye_img,
                result,
                np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
                (330, 220),  # 위치를 왼쪽으로 20만큼 이동
                cv2.MIXED_CLONE
            )

            # 입의 좌표
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
                (260, 320),  # 위치를 왼쪽으로 20만큼 이동
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
        # 스티커 폴더 경로 지정
        self.sticker_folder = 'stickers'  # 스티커 파일이 저장될 폴더
        self.sticker_path = os.path.join(self.sticker_folder, 'sticker1.png')  # 기본 스티커 파일 경로
        
        # 스티커 이미지 로드
        self.img_hat = cv2.imread(self.sticker_path, cv2.IMREAD_UNCHANGED)
        if self.img_hat is None:
            print(f"스티커 이미지 로드 실패: {self.sticker_path}")  # 디버깅 메시지 추가
            # 기본 이미지로 대체하거나 예외 처리
            self.img_hat = np.zeros((100, 100, 4), dtype=np.uint8)  # 기본 빈 이미지 생성

        # 모델 설정
        self.detector_hog = dlib.get_frontal_face_detector()
        self.model_path = 'shape_predictor_68_face_landmarks.dat'
        self.landmark_predictor = dlib.shape_predictor(self.model_path)

        # 스티커 크기 조절 변수
        self.scaling_factor_width = 1.0  # 기본 가 크기
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

            # 경 체크 및 조정
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

        # 과 프레임 반환
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

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

@app.route('/set_sticker', methods=['POST'])
def set_sticker():
    sticker_index = request.form['sticker_index']  # 스티커 인덱스 받기
    file = request.files['file']  # 업로드된 파일 받기
    if file:
        # 스티커 파일을 stickers 폴더에 저장
        file_path = os.path.join('stickers', f'sticker{sticker_index}.png')  # 스티커 파일 경로
        file.save(file_path)  # 파일 저장
        print(f"파일 저장됨: {file_path}")  # 디버깅 메시지 추가
    return '', 204  # 응답 없이 상태 코드 204 반환

# 스티커 변경 기능을 위한 라우트 추가
@app.route('/change_sticker/<int:sticker_index>', methods=['POST'])
def change_sticker(sticker_index):
    global camera
    camera.sticker_path = f'stickers/sticker{sticker_index}.png'  # 스티커 경로 변경
    print(f"변경할 스티커 경로: {camera.sticker_path}")  # 디버깅 메시지 추가
    camera.img_hat = cv2.imread(camera.sticker_path, cv2.IMREAD_UNCHANGED)  # 스티커 이미지 업데이트
    if camera.img_hat is None:
        print(f"스티커 이미지 로드 실패: {camera.sticker_path}")  # 디버깅 메시지 추가
    return '', 204  # 응답 없이 상태 코드 204 

@app.route('/delete_sticker/<int:sticker_index>', methods=['DELETE'])
def delete_sticker(sticker_index):
    file_path = os.path.join('stickers', f'sticker{sticker_index}.png')  # 삭제할 스티커 파일 경로
    try:
        os.remove(file_path)  # 파일 삭제
        print(f"파일 삭제됨: {file_path}")  # 디버깅 메시지
        return '', 204  # 응답 없이 상태 코드 204 반환
    except FileNotFoundError:
        print(f"파일을 찾을 수 없음: {file_path}")  # 파일이 없을 경우 메시지
        return '', 404  # 파일이 없을 경우 상태 코드 404 반환
    
@app.route('/stickers/<int:sticker_index>.png')
def get_sticker(sticker_index):
    # 스티커 파일 경로 설정
    file_path = os.path.join('stickers', f'sticker{sticker_index}.png')
    if os.path.exists(file_path):
        return send_from_directory('stickers', f'sticker{sticker_index}.png')
    return '', 404  # 파일이 없으면 404 반환

@app.route('/adjust_sticker_position', methods=['POST'])
def adjust_sticker_position():
    data = request.get_json()
    global camera
    camera.sticker_offset_x += data['offsetX']
    camera.sticker_offset_y += data['offsetY']
    return '', 204

@app.route('/adjust_sticker_size', methods=['POST'])
def adjust_sticker_size():
    data = request.get_json()
    global camera
    if data['dimension'] == 'width':
        camera.scaling_factor_width += data['change'] / 100.0  # 비율로 조정
    elif data['dimension'] == 'height':
        camera.scaling_factor_height += data['change'] / 100.0  # 비율로 조정
    return '', 204

@app.route('/reset_sticker', methods=['POST'])
def reset_sticker():
    global camera
    # 초기 상태로 되돌리기
    camera.sticker_offset_x = 0
    camera.sticker_offset_y = 0
    camera.scaling_factor_width = 1.0  # 기본 가로 크기
    camera.scaling_factor_height = 1.0  # 기본 세로 크기
    return '', 204



def generate_frames4():
    cap = cv2.VideoCapture(0)
    img = cv2.imread("ckdmsdn.jpg")
    landmarks_points1 = []
    landmarks_points2 = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    indexes_triangles = []

    # Face 1
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

    while True:
        success, img2 = cap.read()
        if not success:
            break
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(img2)

        # Face 2
        faces2 = detector(img2_gray)
        for face in faces2:
            landmarks = predictor(img2_gray, face)
            landmarks_points2 = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points2.append((x, y))

            points2 = np.array(landmarks_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)

        # Triangulation of both faces
        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            tr1_pt1 = landmarks_points[triangle_index[0]]
            tr1_pt2 = landmarks_points[triangle_index[1]]
            tr1_pt3 = landmarks_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)

            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                               [tr1_pt2[0] - x, tr1_pt2[1] - y],
                               [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            # Triangulation of second face
            tr2_pt1 = landmarks_points2[triangle_index[0]]
            tr2_pt2 = landmarks_points2[triangle_index[1]]
            tr2_pt3 = landmarks_points2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2

            cropped_tr2_mask = np.zeros((h, w), np.uint8)

            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing destination face
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

            # Ensure mask is of type uint8
            mask_triangles_designed = mask_triangles_designed.astype(np.uint8)

            # Check sizes
            if warped_triangle.shape[:2] != mask_triangles_designed.shape[:2]:
                mask_triangles_designed = cv2.resize(mask_triangles_designed, (warped_triangle.shape[1], warped_triangle.shape[0]))

            # warped_triangle을 비트 연산으로 결합
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

        # Face swapped (putting 1st face into 2nd face)
        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)

        # 프레임을 JPEG 형식으로 인코딩하여 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               cv2.imencode('.jpg', result)[1].tobytes() + 
               b'\r\n')
        

@app.route('/video_feed4')
def video_feed4():
    return Response(generate_frames4(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed5')
# def video_feed5():
#     return Response(generate_frames5(), mimetype='multipart/x-mixed-replace; boundary=frame')

    

camera = VideoCamera()



if __name__ == '__main__':
    app.run(debug=True)

