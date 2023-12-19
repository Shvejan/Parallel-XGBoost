import time
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Callback
from xgboost_ray import RayDMatrix, RayParams, train
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Start Ray
ray.init(ignore_reinit_error=True)

class PlottingCallback(Callback):
    def on_trial_complete(self, iteration, trials, trial, **info):
        # This function gets called when a trial is completed
        # You can access the trial's results here and create custom plots
        # For instance, to plot the mean accuracy of all trials at each iteration:
        results = [t.last_result for t in trials if t.last_result is not None]
        accuracies = [result["accuracy"] for result in results]
        plt.plot(accuracies)
        plt.xlabel('Iterations')
        plt.ylabel('Mean Accuracy')
        plt.title('Mean Accuracy over Iterations for all Trials')
        plt.savefig(f"plot_{iteration}.png")
        plt.close()


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

# Define the search space
search_space = {
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

# Training function that integrates with Ray Tune
def train_cifar(config):
    train_set = RayDMatrix(train_x, train_y)
    val_set = RayDMatrix(val_x, val_y)
    evals_result = {}

    bst = train(
        config,
        train_set,
        evals_result=evals_result,
        evals=[(train_set, "train"), (val_set, "val")],
        verbose_eval=False,
        num_boost_round=10,
        early_stopping_rounds=10,
        callbacks=[TuneReportCheckpointCallback(filename="model.xgb")]
    )

# Run the hyperparameter search
analysis = tune.run(
    train_cifar,
    resources_per_trial=RayParams(
        num_actors=1,
        cpus_per_actor=1
    ),
    config=search_space,
    num_samples=10,  # Number of different hyperparameter samples to try
    scheduler=ASHAScheduler(),
    metric="val-merror",
    mode="min",
    callbacks=[PlottingCallback()],
)

# Get the best hyperparameters
best_hyperparameters = analysis.get_best_config(metric="val-merror", mode="min")
print("Best hyperparameters found were: ", best_hyperparameters)

ray.shutdown()
