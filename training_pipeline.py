from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error


class TrainPipeline:
    def __init__(self, model):
        self.model = model

    def perform_gridsearch_and_crossval(self, param_grid, x_train, y_train, fold_method='kfold', n_splits=5, verbose=2, n_jobs=-1):
        """
        Performs a GridSearchCV with k-fold cross-validation on the given model and parameter grid.
        
        Parameters:
        - param_grid: A dictionary with hyperparameters to test.
        - x_train: The training features.
        - y_train: The training labels.
        - fold_method: The method for creating folds ('stratified' or 'kfold').
        - n_splits: Number of splits for cross-validation.
        - verbose: Verbosity level for GridSearchCV.
        - n_jobs: Number of jobs to run in parallel for GridSearchCV.

        Returns:
        - The best model found by GridSearchCV.
        - Cross-validation scores.
        """
        if fold_method == 'stratified':
            fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif fold_method == 'kfold':
            fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else: 
            raise ValueError("fold_method must be either 'stratified' or 'kfold'")
        
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=fold, n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(x_train, y_train)
        self.model = grid_search.best_estimator_
        cv_results = grid_search.cv_results_
        cv_scores = cv_results['mean_test_score']
        return self.model, cv_scores

    def compute_cls_scores(self, x_test, y_test):
        """
        Function to compute the scores of the classification model on the test set.
        
        Parameters:
        - x_test: The test features.
        - y_test: The test labels.
        
        Returns:
        - A dictionary of classification scores.
        """
        y_pred = self.model.predict(x_test)
        scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        for score_name, score_value in scores.items():
            print(f"{score_name.capitalize()}: {score_value}")
        return scores
        
    def compute_reg_scores(self, x_test, y_test):
        """
        Function to compute the scores of the regression model on the test set.
        
        Parameters:
        - x_test: The test features.
        - y_test: The test labels.
        
        Returns:
        - A dictionary of regression scores.
        """
        y_pred = self.model.predict(x_test)
        scores = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        for score_name, score_value in scores.items():
            print(f"{score_name.capitalize().replace('_', ' ')}: {score_value}")
        return scores

    def compute_scores(self, x_test, y_test):
        """
        Computes and prints the appropriate scores based on the type of model.
        
        Parameters:
        - x_test: The test features.
        - y_test: The test labels.
        
        Returns:
        - A dictionary of the computed scores.
        """
        if hasattr(self.model, 'predict_proba'):  # Assuming classification if `predict_proba` is available
            return self.compute_cls_scores(x_test, y_test)
        else:
            return self.compute_reg_scores(x_test, y_test)
        
    def plot_model_comparison(self, cv_scores_dict):
        """
        Plots a bar plot to compare the accuracy of different models.
        
        Parameters:
        - cv_scores_dict: A dictionary where keys are model names and values are lists of cross-validation scores.
        """
        model_names = list(cv_scores_dict.keys())
        means = [np.mean(cv_scores_dict[model]) for model in model_names]
        stds = [np.std(cv_scores_dict[model]) for model in model_names]
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, means, yerr=stds, capsize=5)
        plt.title('Model Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.show()
