# 🧠 Deep Learning Image Classifier — CIFAR-10 with TensorFlow CNN

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-API-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-metrics-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/CodeTech-Internship%20Project%202-blueviolet?style=for-the-badge"/>
</p>

<p align="center">
  <b>A complete end-to-end Convolutional Neural Network pipeline for multi-class image classification — built as Project 2 of the CodeTech IT Solutions Internship.</b>
</p>

---

## 📖 The Story Behind This Project

Every great project starts with a question. Mine started with a simple one:

> *"Can a machine really look at a tiny 32×32 image and tell you whether it's a frog or an airplane?"*

While exploring deep learning resources during my CodeTech internship, I kept running into one dataset that every researcher, every tutorial, every benchmark seemed to revolve around — **CIFAR-10**. Originally assembled by the **Canadian Institute For Advanced Research**, this dataset had become the _Hello World_ of visual intelligence. 10 classes. 60,000 images. Decades of research built on top of it.

I didn't want to just use it. I wanted to understand it from scratch — build a model that sees raw pixels and learns, layer by layer, what makes a horse a horse and a truck a truck. No pretrained weights. No shortcuts. Just a CNN trained from the ground up.

What followed was a full pipeline: data loading, normalization, model architecture design, training, evaluation, visualization, and finally — a saved model ready for deployment. This README documents all of it.

---

## 📁 Project Structure

```
📦 cifar10-cnn-classifier/
├── 📄 main.py                          # Main script — the entire pipeline
├── 🖼️  cifar10_sample_images.png       # First 10 training images visualized
├── 📊 training_accuracy.png            # Training vs Validation accuracy plot
├── 📊 training_loss.png                # Training vs Validation loss plot
├── 🔍 sample_predictions.png           # Model predictions on test images (green=correct, red=wrong)
├── 🔢 confusion_matrix.png             # Full 10-class confusion matrix
├── 🤖 cnn_model.h5                     # Saved trained model
└── 📄 README.md                        # You're reading this
```

---

## 🎯 What This Project Demonstrates

| Capability | Detail |
|---|---|
| **Data Handling** | Loading, splitting, and normalizing image datasets |
| **CNN Architecture** | Multi-layer Conv2D + MaxPooling + Dense network design |
| **Model Training** | Batch training with validation split and epoch tracking |
| **Evaluation** | Accuracy/Loss metrics on held-out test data |
| **Visualization** | Training curves, prediction samples, confusion matrix |
| **Model Persistence** | Saving trained model as `.h5` for future inference |

---

## 🗂️ Dataset — CIFAR-10

The **CIFAR-10** dataset contains **60,000 color images** (32×32 pixels, RGB) evenly distributed across **10 classes**:

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | ✈️ Airplane | 5 | 🐕 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚚 Truck |

- **Training set:** 50,000 images (full set used for training)
- **Test set:** 10,000 images (full set used for evaluation)
- **Source:** Auto-downloaded via `tf.keras.datasets.cifar10`

---

## 🏗️ Model Architecture

The CNN is built using the TensorFlow Keras Sequential API:

```python
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),   # Feature extraction — 32 filters
    layers.MaxPooling2D((2, 2)),                      # Spatial downsampling

    layers.Conv2D(64, (3, 3), activation='relu'),   # Deeper feature maps — 64 filters
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),   # Final convolutional layer

    layers.Flatten(),                                 # Convert 3D → 1D
    layers.Dense(64, activation='relu'),              # Fully connected layer
    layers.Dense(10, activation='softmax')            # Output: probability over 10 classes
])
```

**Architecture Summary:**

```
Layer (type)             Output Shape          Param #
─────────────────────────────────────────────────────
Conv2D (32 filters)      (None, 30, 30, 32)    896
MaxPooling2D             (None, 15, 15, 32)    0
Conv2D (64 filters)      (None, 13, 13, 64)    18,496
MaxPooling2D             (None, 6, 6, 64)      0
Conv2D (64 filters)      (None, 4, 4, 64)      36,928
Flatten                  (None, 1024)           0
Dense (64 units)         (None, 64)             65,600
Dense (10 units)         (None, 10)             650
─────────────────────────────────────────────────────
Total params: 122,570
```

---

## ⚙️ Pipeline Walkthrough

### Step 1 — Load & Normalize Data

```python
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values from [0, 255] → [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0
```

Pixel normalization ensures numerical stability during training and helps gradient descent converge faster.

---

### Step 2 — Visualize Sample Images

```python
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.savefig('cifar10_sample_images.png')
```

This gives a sanity check — you see the first 10 training images with their true class labels before any training begins.

---

### Step 3 — Compile & Train

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.2,
    batch_size=64
)
```

- **Optimizer:** Adam — adaptive learning rate, industry default
- **Loss Function:** Sparse Categorical Crossentropy — designed for integer class labels
- **Batch Size:** 64 — balances speed and gradient stability
- **Validation Split:** 20% of training data held back for live monitoring

---

### Step 4 — Plot Training Curves

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

Training and validation accuracy/loss are plotted across epochs. These curves reveal whether the model is learning, overfitting, or underfitting.

---

### Step 5 — Evaluate on Test Set

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
```

The model is evaluated on the **full 10,000-image test set** — data it has never seen during training.

---

### Step 6 — Visualize Predictions

```python
for i in range(12):
    true_label = class_names[y_test[i][0]]
    pred_label = class_names[predicted_labels[i]]
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"P: {pred_label}\nT: {true_label}", color=color)
```

12 test images are shown with predicted vs true labels. **Green = correct**, **Red = incorrect**. This makes model errors immediately interpretable.

---

### Step 7 — Confusion Matrix

```python
cm = confusion_matrix(y_test.flatten(), predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation=45)
```

A full **10×10 confusion matrix** shows exactly which classes the model confuses with each other (e.g., cats vs dogs, automobiles vs trucks). This is the most informative evaluation artifact beyond simple accuracy.

---

### Step 8 — Save the Model

```python
model.save("cnn_model.h5")
```

The trained model is saved in Keras `.h5` format and can be reloaded anytime for inference without retraining.

---

## 🚀 How to Run This Project

### Prerequisites

Make sure you have Python 3.8+ installed. Then install the required libraries:

```bash
pip install tensorflow matplotlib numpy scikit-learn
```

### Clone the Repository

```bash
git clone https://github.com/Adikun2007/codetech-project-2-DL_Image_Classification_Using_TensorFlow.git
cd codetech-project-2-DL_Image_Classification_Using_TensorFlow
```

### Run the Script

```bash
python main.py
```

The CIFAR-10 dataset will be **automatically downloaded** by TensorFlow on first run (~170 MB). After that it's cached locally.

### What to Expect

```text
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
✔ Dataset loaded and normalized

Epoch 1/10
625/625 [==============================] - Training begins...
...
Epoch 10/10
625/625 [==============================] - Training completed

313/313 - Test Accuracy: ~0.65–0.75 (10 epochs on the full CIFAR-10 training set)

✔ cifar10_sample_images.png saved
✔ training_accuracy.png saved
✔ training_loss.png saved
✔ sample_predictions.png saved
✔ confusion_matrix.png saved
✔ cnn_model.h5 saved

Model saved as cnn_model.h5
All visualization images have been saved successfully.
```

> **Note:** This project trains on the complete CIFAR-10 dataset (50,000 training images and 10,000 test images) for 10 epochs. The model typically achieves approximately 65–75% test accuracy, and increasing model complexity or training for more epochs can further improve performance.

---

## 📊 Expected Output Files

| File | Description |
|---|---|
| `cifar10_sample_images.png` | First 10 training images with class labels |
| `training_accuracy.png` | Accuracy curves across epochs |
| `training_loss.png` | Loss curves across epochs |
| `sample_predictions.png` | 12 test image predictions (green/red labels) |
| `confusion_matrix.png` | Full 10×10 class confusion matrix |
| `cnn_model.h5` | Saved Keras model for reuse |

---

## 🔧 How to Improve Performance

This project is intentionally kept minimal and clean to demonstrate the core pipeline. Here's how to push accuracy higher:

```python

# 1. Add Dropout to reduce overfitting
layers.Dropout(0.4)

# 2. Add Batch Normalization for faster convergence
layers.BatchNormalization()

# 3. Add Data Augmentation
tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1
)
```

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **TensorFlow 2.x** | Deep learning framework |
| **Keras API** | Model building and training interface |
| **NumPy** | Numerical array operations |
| **Matplotlib** | Training curve and prediction visualization |
| **scikit-learn** | Confusion matrix computation and display |
| **CIFAR-10** | Benchmark image classification dataset |

---

## 🧑‍💻 About This Project

This is **Project 2** of my Deep Learning internship at **CodeTech IT Solutions**.

The goal was to implement an image classification model using TensorFlow — from raw data ingestion to a fully saved, evaluated model. Rather than using a pretrained model, I deliberately built and trained a CNN from scratch to understand every layer of the pipeline: what Conv2D actually does to an image, why MaxPooling helps, and how a final softmax layer turns 64 numbers into a class prediction.

The choice of CIFAR-10 was driven by its ubiquity as a benchmark — it's small enough to train locally in minutes, yet complex enough that naive approaches fail. It's the dataset that separates models that merely memorize from those that genuinely generalize.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with 🧠 during the <b>CodeTech IT Solutions Internship</b> · Deep Learning Track
</p>
