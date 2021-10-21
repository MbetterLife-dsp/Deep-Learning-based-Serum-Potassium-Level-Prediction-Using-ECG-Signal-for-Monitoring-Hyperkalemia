# Deep-Learning-based-Serum-Potassium-Level-Prediction-Using-ECG-Signal-for-Monitoring-Hyperkalemia

This repository is part of my master degree thesis.

Detailed contents can be confirmed in the pdf file of this repository.

# Background

### Hyperkalemia

* Hyperkalemia can affect the heart rhythm, which can eventually lead to heart failure.
* Hyperkalemia is defined when SPL is above 5.5mEq/L.
* Hyperkalemia is a clinical abnormality frequently seen in patients with chronic kidney disease (CKD).
  * Increased number of CKD patients.

  ![image](https://user-images.githubusercontent.com/86009768/137897151-15459392-05df-41a3-b1eb-845aa32164f8.png)

### ECG Changes in Hyperkalemia

* The SPL causes deformities of the ECG pattern
  * ECG changes of hyperkalemia
  
    ![image](https://user-images.githubusercontent.com/86009768/137899982-5594649c-3638-494d-99f0-c0465c72b2db.png)

### Purpose of this research

* Patients with CKD often do not detect abnormalities in the heart rhythm until they reach the emergency room.
* Deep learning model for SPL prediction using ECG is proposed.
* Compact deep learning model can warn hyperkalemia to CKD patients in compact system.

### Electrocardiogram (ECG) input to the deep learning model

1. Digitalization process of ECG 
  * 1-dimension ECG waves are extracted through image processing from 12-lead ECG images measured in hospital machines.
  
  ![image](https://user-images.githubusercontent.com/86009768/138120927-23e66fe6-edd2-4f7d-abfc-49fb3afd36c0.png)
  
2. Digitalization process of ECG   
  * ECG morphologies are detected for extraction of one-cycle ECG.
  * In order to observe changes in the ECG dependent to hyperkalemia, normalization and synchronization of R-peak are performed. 
  
   ![image](https://user-images.githubusercontent.com/86009768/138121349-938bc128-dd87-477f-8c6e-cdc3c27b067b.png)

### Deep learning model for serum potassium level prediction

* Depthwise separable convolutional neural network (DSCNN) is method to build light weight deep neural network.
* DSCNN factorize a standard convolution into a depthwise convolution and pointwise convolution.
  * Comparison of structure
   ![image](https://user-images.githubusercontent.com/86009768/138124973-b8092ed2-80c6-49df-9422-31e07498b319.png)


  * Comparison of total parameters
  ![image](https://user-images.githubusercontent.com/86009768/138125014-37bf5471-0163-4c48-be4b-467e6f39edad.png)

* Deep Learning Model for SPL Prediction
* We adapted CRNN model based on a depthwise separable convolution kernel together with LSTM.
  * Depthwise separable convolutional recurrent neural network (DSCRNN) identify morphological deformations of ECG.
  * DSCRNN reduce the trainable parameters of CRNN.
    * Structure of deep learning model for SPL prediction.
      ![image](https://user-images.githubusercontent.com/86009768/138125283-e0d7b395-0a40-4d77-8336-cd7aee1a3be8.png)

# Experiment results

* Datasets
  * ECGs from CKD patients with an average age of 73 years were recorded at Wonju Severance Christian Hospital from 2009 to 2019.
  * 1,879 ECGs of the CKD patients were used for the experiment.
    * Summary of the ECG datasets of CKD patients
     ![image](https://user-images.githubusercontent.com/86009768/138126983-72638c70-5a8d-45ef-b1f0-a058755eb2a9.png)

* Experiment for ECG Lead Selection
  * Precordial leads : V1, V2, V3, V4, V5 and V6 were used for SPL prediction performance
    * For performance evaluation, 5-fold cross validation was used.
    * The results show that V5-lead has the highest prediction performance.
      * Scatter-plot of SPLs prediction performance of precordial leads
        ![image](https://user-images.githubusercontent.com/86009768/138127293-9aae81e7-0170-4580-b9c6-d853999e4acd.png)
        
* Experiment for One-Cycle ECG
  * Comparison of one-cycle ECG and full ECG (2 seconds long)
    * Example of 2-seconds
    
      ![image](https://user-images.githubusercontent.com/86009768/138127424-a7ba8211-9d19-4b8a-8b37-e972142a713b.png)
    
    * Learning curves of the proposed deep learning model
      
      ![image](https://user-images.githubusercontent.com/86009768/138127466-8ec81a2c-27d1-4254-aa0b-a0681d447e85.png)

* SPL Prediction Performance Comparison
  * We compared SPLs prediction performances of FNN, CNN, RNN, CRNN and DSCRNN networks to verify the benefits of DSCRNN network.
  * We also compare the model for the detection of hyperkalemia. (C. D. Galloway et al., 2012)
    ![image](https://user-images.githubusercontent.com/86009768/138127554-1d563a27-a28d-46fc-8063-177d4920348d.png)

* Results of SPL prediction
  * Box-plots of SPL prediction performance of DSCRNN model for each trainable parameter
    (a) 180k, (b) 10k, (c) 4k
    
    ![image](https://user-images.githubusercontent.com/86009768/138127674-a514e86e-cbd1-488d-9c0d-ca78c1ae0edf.png)

  * Result shows that the DSCRNN – model maintains relatively high SPL prediction performance even in low trainable parameters.



