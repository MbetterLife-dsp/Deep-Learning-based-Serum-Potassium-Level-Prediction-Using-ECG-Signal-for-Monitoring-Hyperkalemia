# Deep-Learning-based-Serum-Potassium-Level-Prediction-Using-ECG-Signal-for-Monitoring-Hyperkalemia

These repository is part of my master degree thesis.

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

* Digitalization process of ECG (부가설명필요)
  
  ![image](https://user-images.githubusercontent.com/86009768/138120927-23e66fe6-edd2-4f7d-abfc-49fb3afd36c0.png)
  ![image](https://user-images.githubusercontent.com/86009768/138121349-938bc128-dd87-477f-8c6e-cdc3c27b067b.png)

### Deep learning model for serum potassium level prediction

* Depthwise separable convolutional neural network (DSCNN) is method to build light weight deep neural network.
* DSCNN factorize a standard convolution into a depthwise convolution and pointwise convolution.
 * Comparison of structure
   ![image](https://user-images.githubusercontent.com/86009768/138124973-b8092ed2-80c6-49df-9422-31e07498b319.png)


 * Comparison of total parameters
  ![image](https://user-images.githubusercontent.com/86009768/138125014-37bf5471-0163-4c48-be4b-467e6f39edad.png)

