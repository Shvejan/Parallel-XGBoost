import ray
from ray.data.datasource import CIFAR10Datasource

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Download CIFAR-10 dataset
ds = ray.data.read_datasource(CIFAR10Datasource(), parallelism=4)

# Load the dataset into RayDMatrix
dmatrix = ds.to_ray_dmatrix(label_column="label")

dmatrix

