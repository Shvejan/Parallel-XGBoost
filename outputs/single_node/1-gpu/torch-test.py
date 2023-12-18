import time
from xgboost_ray import RayDMatrix, RayParams, train
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os

gpu_count = torch.cuda.device_count()
gpu_name = []

for i in range(gpu_count):
    gpu_name.append(torch.cuda.get_device_name(i))

cpu_count = os.cpu_count()

print(f"Number of GPUs: {gpu_count}")
print("GPU Names:")

for i, name in enumerate(gpu_name):
    print(f"  GPU {i + 1}: {name}")

print(f"Number of CPUs: {cpu_count}")
start_time = time.time()

# Load CIFAR-10 dataset
cifar10_train = CIFAR10(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
cifar10_test = CIFAR10(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

# Data loading time
data_loading_time = time.time() - start_time
print(f"Data loading time: {data_loading_time:.2f} seconds")

# Convert CIFAR-10 dataset to numpy arrays for XGBoost
train_x = np.array([np.array(image).flatten() for image, _ in cifar10_train])
train_y = np.array([label for _, label in cifar10_train])
test_x = np.array([np.array(image).flatten() for image, _ in cifar10_test])
test_y = np.array([label for _, label in cifar10_test])

# Preprocessing time
preprocessing_time = time.time() - start_time - data_loading_time
print(f"Preprocessing time: {preprocessing_time:.2f} seconds")

# Split into training and validation sets
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.1, random_state=42
)

# Create RayDMatrix
train_set = RayDMatrix(train_x, train_y)
val_set = RayDMatrix(val_x, val_y)

# Initialize the dictionary to store evaluation results
evals_result = {}

# Start training time measurement
training_start_time = time.time()

# Train the model
bst = train(
    {
        "objective": "multi:softmax",
        "eval_metric": ["mlogloss", "merror"],
        "num_class": 10,
        "seed": 42,
        "tree_method": "hist",
        "device": "cuda",
    },
    train_set,
    evals_result=evals_result,
    evals=[(train_set, "train"), (val_set, "val")],
    verbose_eval=True,
    num_boost_round=10,
    early_stopping_rounds=10,
    ray_params=RayParams(num_actors=1, gpus_per_actor=1, cpus_per_actor=8),
)

# Training time
training_time = time.time() - training_start_time
print(f"Training time: {training_time:.2f} seconds")

# Save the model
bst.save_model("cifar10_model.xgb")

# Extract the evaluation metrics
train_errors = evals_result["train"]["merror"]
val_errors = evals_result["val"]["merror"]
epochs = len(train_errors)

# Calculate accuracy from error
train_accuracy = [1 - err for err in train_errors]
val_accuracy = [1 - err for err in val_errors]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_accuracy, label="Train Accuracy")
plt.plot(range(epochs), val_accuracy, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy over Epochs")
plt.legend()
plt.grid(True)

# Save plot to file
plt.savefig("training_validation_accuracy_cpu_only.png")
plt.show()

# Print final accuracy values
print(f"Final training accuracy: {train_accuracy[-1]:.4f}")
print(f"Final validation accuracy: {val_accuracy[-1]:.4f}")

# Total execution time
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")
