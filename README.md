# BikeHelmetsDetec
Bike Helmets Detection

Final project for a Python dev - AI-oriented training @Digital.City

Data was scraped on Google, from scratch
Data was labeled by me through ROBOFLOW. I also used ROBOFLOW to preprocess and augment the data

Exported the final version of the Data on Colab to fine-tune a pre-trained YOLOv8 model 

The repo has several models (nano, small, medium - with different data)
My personal favorite is the Final - YOLOM86e.pt
Choose the one that fits the best your need (speed, mAP, etc.) - in the pt folder
EDIT: I could only add 1 small .pt file (Final - YOLON90e)

Use requirements.txt to install all the libraries

Data link: https://universe.roboflow.com/marcvador/bike-helmets-detection-nwmqk

Image Scraper: https://github.com/ohyicong/Google-Image-Scraper

Tracker: https://github.com/abewley/sort.git
