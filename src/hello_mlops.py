import sys
import pandas as pd
import numpy as np
import sklearn
import mlflow
import dvc

print("=" * 50)
print("MLOps Environment Ready 🚀")
print("=" * 50)

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("sklearn:", sklearn.__version__)
print("mlflow:", mlflow.__version__)
print("dvc:", dvc.__version__) 