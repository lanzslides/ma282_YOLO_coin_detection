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
