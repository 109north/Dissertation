# Tools

## extract_image_prediction.py

This is a custom function build by me that allows users to extract model predictions and corresponding ground truths using command line
arguments to dictate the model to pull from and the image filepaths (in relation to the GPU server) to select. 

## infer_torchvision_frcnn.py

This is the file used for **testing** a model after training. It tests on the 441 CityPersons testing images and outputs recall, preciosin,
and AP. It also outputs a dataframe of filepaths, confidence scores, and IoU's for False Negatives and False Positives.

## train_torchvision_frcnn.py

This is the file used for **training** a model. It trains on the 2500 CityPersons training images, plus any augmented images of which amount
and type are dictated by command line arguments by the user. It outputs the model parameters into the directory and filename dictated in
../config/citypersons.yaml
