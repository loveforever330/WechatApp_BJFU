from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO


app = Flask(__name__)


# 这里是首页
@app.route("/")
def index():
    return render_template("upload.html")


# 这里是保存用户上传的图片到文件夹中
@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        f = request.files["file"]
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(
            basepath, "static/images", secure_filename(f.filename)
        )
        f.save(upload_path)
        print(f)
        names_list = []#名称列表
        confidence_list=[]#置信度列表
        # 下半部分用于yolov8的训练代码
        model = YOLO("best.pt")  # 训练好的识别模型
        results = model.predict(source=upload_path, conf=0.25)
        for result in results:
            classes = result.names  # 类别名称
            confidences = result.boxes.conf  # 置信度
            class_ids = result.boxes.cls  # 类别 ID
            for class_id, confidence in zip(class_ids, confidences):
                class_name = classes[int(class_id)]  # 获取类别名称
                names_list.append(class_name)
                confidence_list.append(confidence) 
        return render_template("upload_success.html", names=names_list,confidences=confidence_list)
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(port=5000)
