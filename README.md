# Bank-Check-Verification_System_using_YOLOV5_CycleGan_and_SiameseModel
Signature verification systems are an essential part of most business practices. A significant amount of time and skillful resources could be saved by automating this process. This project demonstrates the implementation of an end-to-end signature verification system for bank check.

YOLOv5 uses the user selected banknote's check for extraction of signatures. Although this process performs well on documents in the real world, it can encounter difficulties due to noise such as printed text or stamps. A CycleGAN-derived method for cleaning noise is incorporated as well. Afterward, Siamese networks are employed to confirm if the cleaned data is fake or authentic.

The project works in three phases:

- The bank check image is processed using openCV after the user selects it for verification. then YOLOv5 model will be run to identify and crop the check parts present in the image, such as its signature and account number. After this, data from the bank check is extracted using easyocr.(The Yolo model is trained using a custom dataset built from bank check images. The data in this dataset has been annotated using roboflow)

- 









# Project Demo
https://drive.google.com/file/d/13948C5ItkJOHFLWi7awMC8HOD55tf-50/view?usp=share_link
