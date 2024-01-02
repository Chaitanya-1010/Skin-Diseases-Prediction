import os
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

model = tf.keras.models.load_model("model.h5", compile=False)
img_size = (192, 192, 3)
UPLOAD_FOLDER = os.path.join("static")
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict():
    path = "static/input_.jpg"
    class_names = [
        "Acne and Rosacea Photos",
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
        "Atopic Dermatitis Photos",
        "Cellulitis Impetigo and other Bacterial Infections",
        "Eczema Photos",
        "Exanthems and Drug Eruptions",
        "Herpes HPV and other STDs Photos",
        "Light Diseases and Disorders of Pigmentation",
        "Lupus and other Connective Tissue diseases",
        "Melanoma Skin Cancer Nevi and Moles",
        "Poison Ivy Photos and other Contact Dermatitis",
        "Psoriasis pictures Lichen Planus and related diseases",
        "Seborrheic Keratoses and other Benign Tumors",
        "Systemic Disease",
        "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Urticaria Hives",
        "Vascular Tumors",
        "Vasculitis Photos",
        "Warts Molluscum and other Viral Infections",
    ]
    image = np.asarray(transform.resize(io.imread(path), img_size[0:2])[:, :, ::-1])
    # plt.show(image)
    plt.imshow(image)
    prediction = model.predict(image[None, ...])
    prediction = prediction / np.sum(prediction) * 100
    prediction = prediction.tolist()[0]
    d = dict(zip(class_names, prediction))
    sorted_d = dict(sorted(d.items(),key=lambda item: item[1],reverse=True))
    return class_names[np.argmax(prediction)], max(prediction), sorted_d, path
predict()