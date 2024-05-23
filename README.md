# Error Page Detector
[**Online Demo**](https://yancyqin.github.io/errorPageDetector/)

## User Guide
### ML5 Javascript Version
index.html preloads the fine tuned pretrained open source computer vision model mobilenet 
It also provides functions to further fine tune the model.

### Python Version
train.py build a CNN model `image_classifier_model.h5` for image classification tasks, particularly binary classification, where the input images are of size 224x224 with 3 color channels (RGB)
predic.py use the model to classify error and normal pages.

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- TensorFlow
- Pillow
- Matplotlib
- NumPy

You can install these packages using pip:

```sh
pip install tensorflow pillow matplotlib numpy scipy
```

### Reference
- MobileNet: [https://github.com/tensorflow/tfjs-models/tree/master/mobilenet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)
- ml5js: [https://learn.ml5js.org/#/](https://learn.ml5js.org/#/)

