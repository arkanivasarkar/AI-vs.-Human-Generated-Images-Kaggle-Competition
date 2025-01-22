import os
import numpy as np
from numpy import linalg as LA
import h5py
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


class extract_features:
    def __init__(self):
        self.model = EfficientNetB7(weights = 'imagenet', 
              input_shape = ((224, 224, 3)), 
              pooling = 'avg', 
              include_top = False)
        
        
    def image2features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = self.model.predict(img)
        normalized_feature = feature[0]/LA.norm(feature[0])
        return normalized_feature
    

    def extract_and_save_features(self, image_path, label_path, save_location):

        # Read labels
        labels_file = pd.read_csv(label_path)
        filenames = np.array(labels_file['file_name'].values)
        labels = labels_file['label'].values
       
        feature_array = []
        label_array = []
        image_full_file_paths = os.listdir(image_path)

        for i in tqdm(range(len(image_full_file_paths))):
            idx = np.where(filenames == f'train_data/{image_full_file_paths[i]}')
            label_array.append(labels[idx][0])
            feature_array.append(self.image2features(os.path.join(image_path,image_full_file_paths[i])))

        feature_array = np.array(feature_array)
        label_array = np.array(label_array)

        print(" Writing Features")
        h5f = h5py.File(os.path.join(save_location, 'efficientNetB7_features.h5'), 'w')
        h5f.create_dataset('features', data=feature_array)
        h5f.create_dataset('labels', data=label_array) 
        h5f.close()



if __name__ == "__main__":
    image_path = "C:\\Users\\arkaniva\\Downloads\\AI-vs.-Human-Generated-Images-Kaggle-Competition\\train_data"
    label_path = "C:\\Users\\arkaniva\\Downloads\\AI-vs.-Human-Generated-Images-Kaggle-Competition\\train.csv"
    save_location = "C:\\Users\\arkaniva\\Downloads\\AI-vs.-Human-Generated-Images-Kaggle-Competition"
    efficientNet_feature_extractor = extract_features()
    efficientNet_feature_extractor.extract_and_save_features(image_path, label_path, save_location)

