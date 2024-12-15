from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt;
import os
import cv2
import joblib

# Define the parent folder path
print("Loading Data")
parent_folder_path = '../FinalProject/BreastDataset/IDC_regular_ps50_idx5/'
X=[];y=[]
images_0 = 1
# Iterate through each folder from 8863 to 16896
for folder_name in range(8863, 16897):
    folder_path = os.path.join(parent_folder_path, str(folder_name))
    
    # Iterate through subfolders 0 and 1
    for subfolder_name in ['0', '1']:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        
        # Check if the subfolder exists
        if os.path.exists(subfolder_path):
            # Iterate through each image in the subfolder
            for image_name in os.listdir(subfolder_path):
                # Get the full path of the image
                image_path = os.path.join(subfolder_path, image_name)
                
                # Check if it's a file (not a directory)
                if os.path.isfile(image_path):
                    # Append the image to the corresponding array
                    if subfolder_name == '0' and images_0 < 78787:
                        Non_C = cv2.cvtColor(cv2.resize(cv2.imread(image_path), (50,50)), cv2.COLOR_BGR2GRAY)
                        X.append(Non_C)
                        y.append(0)
                        images_0 = images_0 + 1
                    elif subfolder_name == '1':
                        C = cv2.cvtColor(cv2.resize(cv2.imread(image_path), (50,50)), cv2.COLOR_BGR2GRAY)
                        X.append(C)
                        y.append(1)
# Now images_0 contains images from subfolder 0 and images_1 contains images from subfolder 1
X=np.asarray(X)
y=np.asarray(y)


# Set the seed for reproducibility
np.random.seed(42)

# Shuffle the indices of the data
indices = np.arange(len(X))
np.random.shuffle(indices)

# Define the proportion of data to be used for testing
test_size = 0.2

# Calculate the number of samples for testing
test_samples = int(len(X) * test_size)

# Split the data into training and testing sets
X_train = X[indices[:-test_samples]]
y_train = y[indices[:-test_samples]]
X_test = X[indices[-test_samples:]]
y_test = y[indices[-test_samples:]]


X_train_flattened = X_train.reshape(len(X_train), -1)
X_test_flattened = X_test.reshape(len(X_test), -1)

print("Training SVM Classifier...  This could take a while... for SVM model like a long while... clip the data to smaller subset for faster training")
# Create a SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the SVM classifier
svm_classifier.fit(X_train_flattened, y_train)

y_pred = svm_classifier.predict(X_test_flattened)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# You can also print other evaluation metrics like precision, recall, etc.
print(classification_report(y_test, y_pred))


joblib.dump(svm_classifier, 'svm_model.pkl')