**Network Intrusion Detection System Using Deep Learning**

**Table of Content:**

* Introduction
* Dataset Used
* Technologies & Tools Used
* Model Architecture
* How to Run
* Results
* Methodology
* Future Enhancements
* Author

**Introduction:**

This project implements a Network Intrusion Detection System (NIDS) using a hybrid CNN-LSTM deep learning model.It focuses on detecting and classifying various network attacks using the UNSW-NB15 dataset. The goal is to improve intrusion detection accuracy by learning complex patterns in network traffic data.

**Dataset Used:**

Name: UNSW-NB15

Abbreviation: University of New South Wales-Network Benchmark 2015

Features: 49 features + class label

Attacks Covered: DoS, Worms, Fuzzer, Reconnaissance, Backdoor, Shellcode, Normal etc.

**Technologies & Tools Used:**

* Python
* TensorFlow / Keras
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn
* SMOTE / ADASYN (for class balancing)
* Google Colab (for training)

**Model Architecture:**

* CNN Layers for spatial feature extraction
* LSTM Layers for sequential pattern learning
* Attention Mechanism to improve model focus on important features
* Final Dense Layer for multiclass classification

**How to Run:**

* **Clone the repository**

git clone https://github.com/[SoniaPugalendran/Network-Intrusion-Detection-System-Using-DeepLearning](https://github.com/SoniaPugalendran/Network-Intrusion-Detection-System-Using-DeepLearning)

* **Install the required libraries**
   pip install -r requirements.txt
* **Run the main Python file**
   python main.py
* **View the results**

The model will train on the preprocessed UNSW-NB15 dataset and output accuracy, precision, recall, F1-score, and confusion matrix in the console or notebook output.

**Result:**

* Classification Report of Binary

![](data:image/png;base64...)

* Classification Report of Multiclass

![](data:image/png;base64...)

##

## **Methodology:**

* Feature scaling using MinMaxScaler
* One-hot encoding of categorical features
* Resampling using SMOTE and ADASYN
* CNN-LSTM fusion model with attention layer
* Focal loss and class weights to handle class imbalance

**Future Enhancements:**

* Real-time attack detection integration using Flask API
* Model optimization using Bayesian hyperparameter tuning
* Deployment on edge devices (Raspberry Pi)

**Author**

**SoniaPugalendran**

[**SoniaPugalendran/Network-Intrusion-Detection-System-Using-DeepLearning**](https://github.com/SoniaPugalendran/Network-Intrusion-Detection-System-Using-DeepLearning)