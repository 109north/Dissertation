# Data Augmentation for Convolutional Neural Networks on Pedestrian Detection

### School of Mathematics and Statistics, University of St Andrews

#### Narayan Murti, 2025.

Convolutional Neural Networks (CNNs) are a deep learning architecture that can be used for pedestrian detection on images. Data augmentation is the process of artificially expanding the input datasets for these networks to diversify the features of its desired classifications. In this study, I tested the effects of traditional data augmentation on a pedestrian detection dataset called CityPersons, using a popular CNN called Faster R-CNN. I explored how flipping, burring, darkening, shrinking, and AugMix affect the accuracy of the model when tested on unseen data. I observed an increase in certain key accuracy metrics including AP and Recall as a result of my targeted augmentations. However, my results were limited by my computing power and the diversity of augmentation types. I struggled with sporadic validation loss due to my small batch size in accordance with my limited GPU memory. Yet overall, I observed an increase in model accuracy due to traditional data augmentation.

The code for this project is contained in this repository. All files were originally created in Jupyter Lab, then pushed onto this repository. They were then pulled onto the University GPU server and training was run on that system. Any bug fixes and edits would occur on GitHub, then be pulled again onto the GPU.

The full report an be accessed through nam27@st-andrews.ac.uk

## References
Ren S, He K, Girshick RB, Sun J. Faster R-CNN: towards real-time object detection with region proposal networks. CoRR. 2015;abs/1506.01497
