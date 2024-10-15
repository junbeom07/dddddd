from flask import Flask, render_template, send_from_directory, Response
import os
import cv2
import dlib

app = Flask(__name__)

# 정적 파일 경로 설정
app.static_folder = 'assets'
# 얼굴 탐지기 및 랜드마크 예측기 초기화
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


if __name__ == '__main__':
    app.run(debug=True)