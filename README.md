# Usage
Here is some sample code on how to use the classes. For the complete example refer to `example_usage.ipynb`.

## Dataset Downloader
```python
import dataset_downloader as dd

dataset_downloader = dd.DatasetDownloader('https://www.kaggle.com/datasets/aksahaha/crop-recommendation')
dataset_downloader.download_dataset()
```

## Training Pipeline
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import training_pipeline as tp

# Define estimators
estimators = {
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'kNN': KNeighborsClassifier(),
}

# Define parameter grids
param_grids = {
    'DecisionTreeClassifier': {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
    },
    'kNN': {
        'n_neighbors': [3, 5, 7],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }
}

# Collect cross-validation scores
all_cv_scores = {}

for estimator_name in estimators:
    print(f"Running GridSearchCV for {estimator_name}...")
    pipeline = tp.TrainPipeline(estimators[estimator_name])
    best_estimator, cv_scores = pipeline.perform_gridsearch_and_crossval(param_grids[estimator_name], x_train, y_train)
    print(f"Best estimator: {best_estimator}")
    test_scores = pipeline.compute_scores(x_test, y_test)
    all_cv_scores[estimator_name] = cv_scores
    print("\n")

# Plot model comparison
pipeline.plot_model_comparison(all_cv_scores)
```