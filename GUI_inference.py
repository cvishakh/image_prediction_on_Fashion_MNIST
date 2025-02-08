import os
import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image

#Path to model
if getattr(sys, 'frozen', False):  #For running as an executable
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "fashion_mnist_model.tflite")

#Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

#Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


class ImageClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Classifier: Fashion MNIST ')
        self.setGeometry(100, 100, 300, 300)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel('Drag and drop image here', self)
        self.result_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.display_image(file_path)
        self.classify_image(file_path)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def classify_image(self, file_path):
        image = Image.open(file_path).convert('L').resize((28, 28))  #Convert to grayscale & resize
        image = np.array(image, dtype=np.float32) / 255.0  #Normalize
        image = image.reshape(1, 28, 28, 1)

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)

        labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                        "Ankle boot"]
        self.result_label.setText(f'Predicted Result: {labels[predicted_class]}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassifierGUI()
    window.show()
    sys.exit(app.exec_())