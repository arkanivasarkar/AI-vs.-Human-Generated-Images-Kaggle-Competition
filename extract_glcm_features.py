import os
import numpy as np
from numpy import linalg as LA
import h5py
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage.color import rgb2gray
import pandas as pd
from skimage import io, color, transform
from skimage.measure import shannon_entropy
import kagglehub



class extract_features:
    def __init__(self):
        data_cols = []
        column_names = ['contrast', 'dissimilarity', 'homogeneity','energy', 'correlation', 'ASM', 'mean','variance', 'std', 'entropy']
        for i in range(12):
          for colname in column_names:
            data_cols += [f'{colname}{i}']
        data_cols += ['shannon_entropy','labels']
        self.df = pd.DataFrame(columns=tuple(data_cols))

  
    def compute_glcm_features(self, image, dist, ang):
      glcm = graycomatrix(image, distances=[1,3,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
      contrast = graycoprops(glcm, 'contrast')[0, 0]
      dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
      homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
      energy = graycoprops(glcm, 'energy')[0, 0]
      correlation = graycoprops(glcm, 'correlation')[0, 0]
      ASM = graycoprops(glcm, 'ASM')[0, 0]
      mean = graycoprops(glcm, 'mean')[0, 0]
      variance = graycoprops(glcm, 'variance')[0, 0]
      std = graycoprops(glcm, 'std')[0, 0]
      entropy = graycoprops(glcm, 'entropy')[0, 0]
      return contrast, dissimilarity, homogeneity, energy, correlation, ASM, mean, variance, std, entropy, glcm


  
    def image2features(self, img_path, label):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
        distances=[1,3,5]
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
        output = []
        for dist in distances:
          for ang in angles:
            contrast, dissimilarity, homogeneity, energy, correlation, ASM, mean, variance, std, entropy, glcm = self.compute_glcm_features(img, dist, ang)
            output.extend([contrast, dissimilarity, homogeneity, energy, correlation, ASM, mean, variance, std, entropy])

        output.extend([shannon_entropy(img), label])
        self.df.loc[len(self.df )] = output
      


    def extract_and_save_features(self, image_path, label_path):
        labels_file = pd.read_csv(label_path)
        filenames = np.array(labels_file['file_name'].values)
        labels = labels_file['label'].values

        image_full_file_paths = os.listdir(image_path)
        for i in tqdm(range(len(image_full_file_paths))):
            idx = np.where(filenames == f'train_data/{image_full_file_paths[i]}')
            self.image2features(os.path.join(image_path, image_full_file_paths[i]), labels[idx][0])
        return  self.df




if __name__ == "__main__":
    path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
    image_path = f"{path}/train_data"
    label_path = f"{path}/train.csv"
    save_location = "/content"

    # Extract features
    feature_extractor = extract_features()
    glcm_features = feature_extractor.extract_and_save_features(image_path, label_path)
    csv_file = 'glcm_features.csv'
    glcm_features.to_csv(f'{save_location}/{csv_file}', index=False)
