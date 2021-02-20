import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single

# Flask import
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import base64

# Khai bao cong cua server
server_host = "127.0.0.1"
server_port = '8000'
api_key = "12345679aA@#$%^&*"
success_code = 200
success_message = "Success"
failure_code_not_detect = 400
failure_code_not_detect_message = "Not Detection"
failure_code_detect_null = 404
failure_code_detect_null_message = "Detection Null"
system_error_code = 500
system_error_message = "System Error, Try Again"


# Doan ma khoi tao server
app = Flask(__name__)
CORS(app)

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'
# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)
# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 204
Dmin = 122
# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu
model_svm = cv2.ml.SVM_load('svm.xml')

# Ham sap xep contour tu trai sang phai
def sort_contours(cnts, method, Ivehicle):
    reverse = False
    i = 1
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][1] * Ivehicle.shape[1] + b[1][0] * Ivehicle.shape[0] +
                                                      b[1][1] * Ivehicle.shape[1], reverse=False))
    return cnts

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Function run detection
def start_detection_lp(image):
    Ivehicle = image
   
    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

    if (len(LpImg)):

        # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        roi = LpImg[0]

        # Chuyen anh bien so ve gray
        gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)

        # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # Segment kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""

        for c in sort_contours(cont, "left-to-right", Ivehicle):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                if h/roi.shape[0]>=0.3: # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tach so va predict
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num,dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)

                    # Dua vao model SVM
                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result<=9: # Neu la so thi hien thi luon
                        result = str(result)
                    else: #Neu la chu thi chuyen bang ASCII
                        result = chr(result)

                    plate_info +=result

        if plate_info:
            return [success_code, plate_info, ""]
        else:
            return [failure_code_detect_null, "null", failure_code_detect_null_message]
    else:
        return [failure_code_not_detect, "not_detect", failure_code_not_detect_message]

# start_detection_lp("test/s.jpg")
@app.route('/detect', methods=['POST'])
@cross_origin()
def detection_api():
    try:
        image_path_test = "test/2323.jpg"
        # image = cv2.imread(image_path_test)

        # Authorization here, check api_key

        # Get base64 image from request
        # image_b64 = request.form.get('image')

        # Test base64 image
        image1 = cv2.imread(image_path_test)
        base64_image = base64.b64encode(cv2.imencode(".jpg", image1)[1]).decode()

        # Decode base64 image to image
        decoded_data = base64.b64decode(base64_image)
        np_data = np.fromstring(decoded_data,np.uint8)

        image = cv2.imdecode(np_data, cv2.IMREAD_ANYCOLOR)

        data = start_detection_lp(image)
        
        return jsonify({
            "code": data[0],
            "data": data[1],
            "message": data[2]
        })
    except:
        return jsonify({
            "code": system_error_code,
            "data": "",
            "message": system_error_message
        })


# Thuc thi server
if __name__ == '__main__':
    app.run(debug = True, host = server_host, port = server_port)