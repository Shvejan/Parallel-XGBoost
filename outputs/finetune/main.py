import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from xgboost_ray import RayDMatrix, RayParams, train
from ray import tune, put
import numpy as np
import ray
import os
import matplotlib.pyplot as plt
import os
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

num_actors = 2
num_cpus_per_actor = 1
ray_params = RayParams(num_actors=num_actors, cpus_per_actor=num_cpus_per_actor)

# Transform for the CIFAR-10 data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)

trainloader_full = torch.utils.data.DataLoader(trainset_full, batch_size=len(trainset_full), shuffle=True)

# Get the data into numpy arrays for XGBoost
images, labels = next(iter(trainloader_full))
images = images.numpy().astype(np.float16)  # Convert to float16
labels = labels.numpy()

# Flatten the image data
images = images.reshape(images.shape[0], -1)

# Split the data into two halves
half_x, _, half_y, _ = train_test_split(images, labels, train_size=0.5, random_state=42)

# Further split each half into training and validation sets (50% of each half)
train_x, val_x, train_y, val_y = train_test_split(half_x, half_y, train_size=0.5, random_state=42)

# Put the training and validation data in the Ray object store
train_x_id = put(train_x)
train_y_id = put(train_y)
val_x_id = put(val_x)
val_y_id = put(val_y)

def train_model(config, train_x_id, train_y_id, val_x_id, val_y_id):
    # Retrieve the data from the object store
    train_x = ray.get(train_x_id)
    train_y = ray.get(train_y_id)
    val_x = ray.get(val_x_id)
    val_y = ray.get(val_y_id)

    train_set = RayDMatrix(train_x, train_y)
    val_set = RayDMatrix(val_x, val_y)

    evals_result = {}
    bst = train(
        params=config,
        dtrain=train_set,
        evals=[(train_set, "train"), (val_set, "eval")],
        evals_result=evals_result,
        verbose_eval=False,
        ray_params=ray_params)
    
    # Save the model
    bst.save_model(f"model_{config['eta']:.4f}_{config['max_depth']}.xgb")


    # Report training and validation error
    ray.train.report({'train_merror': evals_result['train']['merror'][-1],
                      'val_merror': evals_result['eval']['merror'][-1],
                      'eval-mlogloss': evals_result['eval']['mlogloss'][-1]
                      })
if __name__ =="__main__":

    config = search_space = {
    "objective": "multi:softmax",
    "eval_metric": ["mlogloss", "merror"],
    "num_class": 10,
    "max_depth": tune.randint(3, 10),
    # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    
    "min_child_weight": tune.choice([1, 2, 3]),
    # Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results
    #  in a leaf node with the sum of instance weight less than min_child_weight, then the building 
    #  process will give up further partitioning.

    "subsample": tune.uniform(0.5, 1.0),
    # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would
    # randomly sample half of the training data prior to growing trees. This will prevent overfitting.
    "eta": tune.loguniform(1e-4, 1e-1),
    "seed": 42,
    "max_bin": tune.choice([256, 512]),
    # The maximum number of bins that feature values are bucketed in. Using more bins
    # gives you more fine-grained splits but may lead to overfitting

    "alpha": tune.choice([0, 0.5, 1]),
    # L1 regularization term on weights. Increasing this value will make model more conservative.
    }


    # Hyperparameter tuning
    analysis = tune.run(
        lambda config: train_model(config, train_x_id, train_y_id, val_x_id, val_y_id),
        config=config,
        metric="mlogloss",
        mode="min",
        num_samples=4,
        resources_per_trial=ray_params.get_tune_resources())


    print("Best hyperparameters", analysis.best_config)
    print("results:",analysis.results.values())
    # Plotting validation accuracies
    val_accuracies = [1 - x['eval-merror'] for x in analysis.results.values()]
    plt.plot(val_accuracies)
    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over Iterations')
    plt.show()
