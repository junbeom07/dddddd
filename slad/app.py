from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# 정적 파일 경로 설정
app.static_folder = 'assets'

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

if __name__ == '__main__':
    app.run(debug=True)