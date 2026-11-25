# Multi-Class Image Classification using CNN

This project implements a custom TensorFlow/Keras convolutional neural network for classifying CIFAR-10 images into ten classes.

## Classes

Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Project Structure

```text
Multi-Class Image Classification CNN/
├── Task_3_CNN_Image_Classification.ipynb
├── Cifar10_cnn_model.keras
├── Accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── sample_predictions.png
├── Project_Report.pdf
└── README.md
```

## Dataset

The notebook loads CIFAR-10 directly through TensorFlow.

| Property | Value |
|---|---:|
| Training images | 50,000 |
| Test images | 10,000 |
| Image dimensions | 32 x 32 x 3 |
| Pixel range | 0-255, normalized to 0-1 |
| Validation split | 20% of training data |
| Maximum epochs | 40 |
| Batch size | 64 |

## Requirements

- Python 3.9 or newer
- TensorFlow
- NumPy
- Matplotlib
- Scikit-Learn

GPU execution is recommended. The notebook also supports CPU execution.

## Running the Notebook

1. Open `Task_3_CNN_Image_Classification.ipynb`.
2. Enable GPU acceleration if available.
3. Run all cells sequentially.
4. TensorFlow downloads CIFAR-10 during the first run if it is not cached.

In Google Colab, upload the notebook and enable a GPU runtime.

## Model Architecture

- Random horizontal flip
- Random rotation up to 10%
- Random zoom up to 10%
- Random horizontal and vertical translation up to 10%
- Conv2D 32 + Batch Normalization
- Conv2D 32 + Max Pooling + Dropout 0.25
- Conv2D 64 + Batch Normalization
- Conv2D 64 + Max Pooling + Dropout 0.30
- Conv2D 128 + Batch Normalization + Max Pooling + Dropout 0.35
- Dense 128 + Batch Normalization + Dropout 0.50
- Dense 10 with softmax

The model is compiled with Adam (`learning_rate=0.001`) and sparse categorical cross-entropy.

## Training

- Early stopping monitors validation loss with patience 7 and restores the best weights.
- `ReduceLROnPlateau` halves the learning rate after three epochs without validation-loss improvement.
- The minimum learning rate is `1e-5`.

## Evaluation Results

The confusion matrix contains 10,000 test examples. The matrix diagonal contains 7,106 correct predictions, corresponding to approximately 71.1% test accuracy.

The learning curves show validation accuracy generally above training accuracy by the end of training, while validation loss trends downward with fluctuations.

## Outputs

- [Saved Keras model](cifar10_cnn_model.keras)
- [Training and validation accuracy](accuracy_curve.png)
- [Training and validation loss](loss_curve.png)
- [Test confusion matrix](confusion_matrix.png)
- [Sample test predictions](sample_predictions.png)
- [Complete PDF report](Project_Report.pdf)

## Loading the Model

```python
import tensorflow as tf

model = tf.keras.models.load_model("cifar10_cnn_model.keras")
predictions = model.predict(test_images)
```

Test inputs must have shape `(batch_size, 32, 32, 3)` and pixel values scaled to the range 0-1.

## Limitations and Future Work

- The model can confuse visually similar or context-overlapping classes.
- Results can be improved with transfer learning, stronger augmentation, regularization, or a deeper architecture.
- Preserve a fixed random seed to improve reproducibility.
- Report test loss and per-class precision, recall, and F1-score for a complete evaluation.
