# README #
Welcome to my senior year project as part of the Oxford 4th year engineering course. The project is on object classification 
and detection in radar images using deep learning techniques.  


Most of the scripts here are experiments done to investigate how our radar image data could be best utilised to build a 
classification and detection pipeline.  


Some experiments:  

(1) SVM to classify rotated and unrotated car image patches.  

(2a) CNN to classify rotated and unrotated car image patches to push up the performance of the SVM in (1).
(2b) CNN to do regression on the car image patches orientations to predict an angle between 0 and 360.

(3) Autoencoder (AEC) to push up the performance of the CNN in (2).  

(4) Sliding window based detector that performs detection using (2) in near real-time on a Macbook Pro.  



There are scripts that evaluate and display the relative performance of the approaches tried. Common metrics used are Precision-Recall curves
and FPPI vs Miss rate curves as referenced in the project report https://tinyurl.com/y7eqbgsd  


N.B. the data is not open source as it was obtained with the help of Oxford robotics institute and thus these scripts can't quite be 
cloned and run on any machine without the necessary data.  


For more information on the project including data acquisition please refer to the report (linked above).


Nihaar Shah
