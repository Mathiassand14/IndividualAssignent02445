import enum
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from joblib import delayed, Parallel, parallel_backend
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class SandLabelEncoder(LabelEncoder):
    def fit(self, y):
        super().fit(y)
        self.classes_ = np.array(self.classes_, dtype=object)  # Ensure classes are object type
        return self
    
    def transform_proba(self, y_proba):
        proba_dicts = []
        for row in y_proba:
            d = defaultdict(lambda: 0.0)
            for i, p in enumerate(row):
                d[self.classes_[i]] = float(p)
            proba_dicts.append(d)
        return proba_dicts
    
    def keep_seen(self, X_test, Y_test) -> Tuple[np.ndarray, np.ndarray]:
        Y_test = np.array(Y_test)
        mask = np.isin(Y_test, self.classes_)
        return X_test[mask], Y_test[mask]

class SandBaseEstimator(BaseEstimator):
    def fit(self, X, y):
        pass
    
    def predict(self, X) -> list:
        pass
    
    def predict_proba(self, X) -> list:
        pass


class Model:
    def __init__(self, model_name: str, model_instance: SandBaseEstimator, params: dict[str, list]):
        self.model_name = model_name
        self.model_instance = model_instance
        self.params = params
    
    def param_grid(self) -> list:
        return list(ParameterGrid(self.params))
    
    def model(self) -> dict[str, SandBaseEstimator]:
        return {"model_name": self.model_name, "model_instance": self.model_instance}


class EvaluateByClass:
    def __init__(self, evaluate_by: callable, probability: bool = False, classification: bool = False):
        self.evaluate_by = evaluate_by
        self.probability = probability
        self.classification = classification


class EvaluateBy(enum.Enum):
    ExactMatch: EvaluateByClass = EvaluateByClass(
        lambda y_true, y_pred: np.mean(y_true == y_pred),
        probability=False,
        classification=True
    )
    
    MeanAbsoluteError: EvaluateByClass = EvaluateByClass(
        lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        probability=False,
        classification=False
    )
    
    MeanSquaredError: EvaluateByClass = EvaluateByClass(
        lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
        probability=False,
        classification=False
    )


class TwoLayer:
    def __init__(
            self, X, Y, outer,
            inner, models: list[Model],
            groups=None,
            n_jobs=-1, verbose=1,
            random_state=0,
            evaluate_by: EvaluateByClass=
            EvaluateByClass(
                lambda y_true, y_pred, y_predict_proba:
                np.mean(np.abs(y_true - y_pred)),
                probability=False,
                classification=False
            ),
            summarize_by: callable=lambda scores: np.mean(scores),
            sort_by: callable=lambda x: x
    ):
        self.X = X
        self.Y = Y
        self.outer = outer
        self.inner = inner
        self.groups = groups
        self.models = models
        self.n_jobs = n_jobs
        self.Verbose = verbose
        self.random_state = random_state
        self.evaluate_by = evaluate_by.evaluate_by
        self.summarize_by = summarize_by
        self.probability = evaluate_by.probability
        self.classification = evaluate_by.classification
        self.sort_by = sort_by
    
    def fit(self):
        outer_scores: list = []
        best_params_per_fold: list = []
        best_model_names: list = []
        best_models: list = []
        
        total_fits = (
                self.outer.get_n_splits(self.X, self.Y, self.groups) *
                sum(
                    len(model.param_grid()) * self.inner.get_n_splits(self.X, self.Y, self.groups)
                    for model in self.models
                )
        )
        
        progress = tqdm(total=total_fits, desc="TwoLayer CV", unit="fit") if self.Verbose >= 1 else None
        
        
        for train_idx, test_idx in self.outer.split(self.X, self.Y, self.groups):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.Y[train_idx], self.Y[test_idx]
            group_train = self.groups[train_idx] if self.groups is not None else None
            
            all_candidates: List[Tuple[Dict[str, SandBaseEstimator], Dict[str, any]]] = []
            for _model in self.models:
                all_candidates.extend([(_model.model(), params) for params in _model.param_grid()])
            
            def evaluate_params(model: Dict[str, SandBaseEstimator], params: Dict[str, any]):
                inner_scores: List[float] = []
                sle = SandLabelEncoder() if self.classification else None
                for inner_train_idx, inner_test_idx in self.inner.split(X_train, y_train, group_train):
                    X_inner_train, X_inner_test = X_train[inner_train_idx], X_train[inner_test_idx]
                    y_inner_train, y_inner_test = y_train[inner_train_idx], y_train[inner_test_idx]
                    
                    if self.classification:
                        sle.fit(y_inner_train)
                        y_inner_train = sle.transform(y_inner_train)
                        #X_inner_test, y_inner_test = sle.keep_seen(X_inner_test, y_inner_test)
                    
                    model_instance: SandBaseEstimator = clone(model['model_instance'])
                    model_instance.set_params(**params)
                    inner_scores.append(
                        self._fit_model_eval(X_inner_test, X_inner_train, model_instance, sle, y_inner_test,
                                             y_inner_train))
                    
                    
                    
                return self.summarize_by(inner_scores), model["model_instance"], params, model["model_name"]
            results = []
            parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
            for result in parallel(delayed(evaluate_params)(_model, params) for _model, params in all_candidates):
                results.append(result)
                #print(f"Evaluated model: {result[3]} with params: {result[2]} and score: {result[0]}")
                if progress is not None:
                    progress.update(len(all_candidates) * self.inner.get_n_splits(self.X, self.Y, self.groups))
            
            
            
            best_score, best_model, best_params, best_model_name = max(results, key=lambda x: self.sort_by(x[0]))
            best_model: SandBaseEstimator = clone(best_model)
            best_model.set_params(**best_params)
            
            if self.classification:
                sle = SandLabelEncoder()
                sle.fit(y_train)
                y_train = sle.transform(y_train)
                y_test_encoded = sle.transform(y_test)
            else:
                sle = None
                y_test_encoded = y_test
            
            y_pred = self._fit_model_eval(X_test, X_train, best_model, sle, y_test_encoded, y_train)
            outer_scores.append(y_pred)
            best_params_per_fold.append(best_params)
            best_model_names.append(best_model_name)
            best_models.append(best_model)
        
        if progress is not None:
            progress.close()
        
        self.outer_scores = outer_scores
        self.best_params_per_fold = best_params_per_fold
        self.best_model_names = best_model_names
        self.best_models = best_models
        return outer_scores
    
    def _fit_model_eval(self, X_test, X_train, model, sle, y_test, y_train):
        model.fit(X_train, y_train)
        
        y_pred_outer = model.predict(X_test)
        if self.classification:
            y_pred_outer = sle.inverse_transform(y_pred_outer)
        if self.probability:
            y_predict_proba_outer = model.predict_proba(X_test)
            if self.classification:
                y_predict_proba_outer = sle.transform_proba(y_predict_proba_outer)
            outer_score = self.evaluate_by(y_test, y_pred_outer, y_predict_proba_outer)
        else:
            outer_score = self.evaluate_by(y_test, y_pred_outer)
        return outer_score
    
    def get_results(self):
        if not hasattr(self, 'outer_scores'):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return {
            "outer_scores": self.outer_scores,
            "best_params_per_fold": self.best_params_per_fold,
            "best_model_names": self.best_model_names,
            "best_models": self.best_models,
            "mean_score": np.mean(self.outer_scores),
            "std_score": np.std(self.outer_scores)
        }
    
    def get_best_models(self):
        return self.best_models
    
    def get_outer_scores(self):
        return self.outer_scores
    
    def get_best_params(self):
        return self.best_params_per_fold
    
    def get_best_model_names(self):
        return self.best_model_names
    
    def get_mean_score(self):
        return np.mean(self.outer_scores)
    
    def get_std_score(self):
        return np.std(self.outer_scores)
    
    def get_scores(self):
        return self.outer_scores
    
    def get_test_sizes(self):
        return [len(self.X[test_idx]) for _, test_idx in self.outer.split(self.X, self.Y, self.groups)]
