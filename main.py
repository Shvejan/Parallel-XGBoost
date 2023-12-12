from xgboost_ray import RayDMatrix, RayParams, train
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
# Load CIFAR-10 dataset
cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
# Convert CIFAR-10 dataset to numpy arrays for XGBoost
train_x = np.array([np.array(image).flatten() for image, _ in cifar10_train])
train_y = np.array([label for _, label in cifar10_train])
test_x = np.array([np.array(image).flatten() for image, _ in cifar10_test])
test_y = np.array([label for _, label in cifar10_test])
# Split into training and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)
# Create RayDMatrix
train_set = RayDMatrix(train_x, train_y)
val_set = RayDMatrix(val_x, val_y)
# Store eval results
evals_result = {}
bst = train(
    {
        "objective": "multi:softmax",
        "eval_metric": ["mlogloss", "merror"],
        "num_class": 10,
        "seed": 42,
    },
    train_set,
    evals_result=evals_result,
    evals=[(train_set, "train"), (val_set, "val")],
    verbose_eval=True,
    num_boost_round=100,  # number of boosting rounds (epochs)
    early_stopping_rounds=10,
    ray_params=RayParams(
        num_actors=2,
        cpus_per_actor=1
    )
)

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
plt.plot(range(epochs), train_accuracy, label='Train Accuracy')
plt.plot(range(epochs), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)

# Save plot to file
plt.savefig('training_validation_accuracy.png')
plt.show()

# Print final accuracy values
print(f"Final training accuracy: {train_accuracy[-1]:.4f}")
print(f"Final validation accuracy: {val_accuracy[-1]:.4f}")