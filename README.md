# Coin Detection and Denomination: A YOLOv8 Nano Experiment
For MATH 282.1 (Special Topics in Data Science: Computer Vision)

December 4, 2025

## Notes
* To facilitate quicker testing of the trained model, a file named `coin_detector_tester.py` can be used. It is in the `app` folder.
* I recommend storing the tester image in `app/testers` for easier access.
* The best model was used (i.e., with lowest validation loss during training), stored in `best_coin_detector_from_training.pt`.
  * YOLOv8 Nano model was used for training.
* Required packages: `pip install ultralytics opencv-python matplotlib pillow`
* Run `python app/coin_detector_tester.py <image_path>`.
* Expected output: Total value detected, number of coins detected, and a breakdown of the coins in the image with confidence scores.

## Quick Demonstration

<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/d573ab98-6f43-41f6-91ba-f17b1e306202" />

Figure 1. Test image.

<br>
<img width="683" height="129" alt="image" src="https://github.com/user-attachments/assets/b43c452e-3983-469e-ab3e-16252ecef9e7" />

Figure 2. Running the python script and its output.

<br>
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/7736639d-b721-4844-ae96-290807c8b535" />

Figure 3. Annotated image as output.

