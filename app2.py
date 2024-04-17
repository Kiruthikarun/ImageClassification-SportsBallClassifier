import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model('E:\python\Sports _ball_classifier\model.h5')

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        self.load_button = tk.Button(root, text="Load Images", command=self.load_images)
        self.load_button.pack()

        self.predict_button = tk.Button(root, text="Classify Images", command=self.classify_images)
        self.predict_button.pack()

        self.image_labels = []  # List to hold image labels
        self.result_labels = []  # List to hold result labels
        self.image_paths = []  # List to hold image paths

    def load_images(self):
        file_paths = filedialog.askopenfilenames()
        if file_paths:
            for file_path in file_paths:
                self.image_paths.append(file_path)
                img = Image.open(file_path)
                img = img.resize((224, 224), Image.BILINEAR)
                img = ImageTk.PhotoImage(img)

                image_label = tk.Label(self.root, image=img)
                image_label.image = img
                image_label.pack()
                self.image_labels.append(image_label)

                result_label = tk.Label(self.root, text="")
                result_label.pack()
                self.result_labels.append(result_label)

    def classify_images(self):
        if self.image_paths:
            for i, image_path in enumerate(self.image_paths):
                img = image.load_img(image_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = img / 255.0

                predictions = model.predict(img)
                predicted_class = np.argmax(predictions)

                class_labels = ['American football','Baseball','Basketball','Billiard ball','Bowling ball','Cricket ball','Football','Golf ball','Hockey ball','Hockey puck', 'Rugby ball','Shuttlecock','Table Tennis Ball','Tennis Ball','Volleyball']

                result = class_labels[predicted_class]
                self.result_labels[i].config(text=f"Predicted class: {result}")
        else:
            messagebox.showerror("Error", "Please load images first.")

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
