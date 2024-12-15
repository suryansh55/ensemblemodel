import cv2;
import joblib


def predict(image_path):
    image = cv2.cvtColor(cv2.resize(cv2.imread(image_path), (50,50)), cv2.COLOR_BGR2GRAY)
    return loaded_random_forest_model.predict(image.flatten().reshape(1, -1))



loaded_random_forest_model = joblib.load('random_forest_model.pkl')



labels = {
    0: 'Non-Cancerous',
    1: 'Cancerous'
}

print(labels[predict('path_to_image.png')[0]])  # Replace 'path_to_image.jpg' with the path to the image you want to predict