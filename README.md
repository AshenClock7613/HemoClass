# Blood Cell Image Classification

This repository contains code and datasets for blood cell classification using machine learning and deep learning techniques. This project was developed as part of the **Advanced Numerical Methods** course.

## Participants
- **Antonio Spedito**
- **Gaia Montalbano**
- **Alessandro Picone**

## Tasks
- **Blood Cell Classification**: Identifying whether a given blood cell image represents a healthy or cancerous leukocyte.

## Models and Techniques Used
- **Image Compression Methods**: 
  - QR Factorization for matrix decomposition.
  - Singular Value Decomposition (SVD) to reduce data dimensionality while preserving key features.
- **Neural Networks for Classification**:
  - Custom-built neural network trained using Conjugate Gradient Descent.
  - MATLAB-based neural network trained using Fletcher-Reeves and classic Gradient Descent.

## Dataset Preprocessing
- Images are converted to grayscale and resized to standard dimensions.
- Feature extraction is performed using SVD.
- Processed images are used to train the neural networks.

## Usage Instructions
To run the classification tasks:
1. **Image Preprocessing**:
   ```matlab
   run preprocc.m
   ```
2. **Apply QR Factorization**:
   ```matlab
   run func_ourQR.m
   ```
3. **Apply SVD Compression**:
   ```matlab
   run func_ourSVD.m
   ```
4. **Train and Evaluate Neural Networks**:
   - Custom Neural Network:
     ```matlab
     run nn_manual.m
     ```
   - MATLAB Tool-Based Neural Network:
     ```matlab
     run nn_matlab.m
     ```

## Results
- SVD-based compression retained significant information, making it more suitable for training neural networks.
- QR Factorization did not preserve the most relevant features for classification.
- The neural networks achieved high accuracy in distinguishing between healthy and cancerous leukocytes.

## License
This project is distributed under the license found in the **LICENSE** file.

---
For more details, refer to the [Classificazione_Immagini_Sangue.pdf](Classificazione_Immagini_Sangue.pdf) file.

