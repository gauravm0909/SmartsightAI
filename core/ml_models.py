"""
SmartSight AI - Predictive Analytics & Machine Learning Engine
Implements enterprise-grade Regression, Classification, and Clustering pipelines.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
)

from core.data_analyzer import DataAnalyzer


class MLPredictiveEngine:
    """Production ML pipeline featuring automated task detection, leakage prevention, and model benchmarking."""

    ALGORITHMS = {
        "regression": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
        ],
        "classification": [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "K-Nearest Neighbors",
        ],
        "clustering": [
            "K-Means Clustering",
        ],
    }

    @classmethod
    def identify_task_type(cls, df: pd.DataFrame, target_col: Optional[str] = None) -> str:
        """Determines whether problem is Regression, Classification, or Clustering."""
        if not target_col or target_col.strip() == "" or target_col.lower() == "none":
            return "clustering"

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        series = df[target_col].dropna()
        n_unique = series.nunique()

        # If string / categorical or boolean -> Classification
        if series.dtype == "object" or series.dtype == "category" or series.dtype == "bool":
            if n_unique > 50:
                raise ValueError(f"Target column '{target_col}' has too many unique classes ({n_unique}) for classification.")
            return "classification"

        # If numeric
        if pd.api.types.is_numeric_dtype(series):
            # If 2 to 10 discrete integer values -> Classification
            if n_unique <= 10 and (series == series.round()).all():
                return "classification"
            return "regression"

        return "classification"

    @classmethod
    def train_and_evaluate(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        algorithm: str = "Random Forest Regressor",
        test_size: float = 0.20,
        random_state: int = 42,
        n_clusters: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes robust train/test evaluation with leak-free preprocessing and feature importances.
        """
        if not feature_cols:
            raise ValueError("At least one feature column must be selected.")

        task_type = cls.identify_task_type(df, target_col)

        # -------------------------------------------------------------
        # 1. CLUSTERING PIPELINE
        # -------------------------------------------------------------
        if task_type == "clustering":
            return cls._train_clustering(df, feature_cols, n_clusters, random_state)

        # -------------------------------------------------------------
        # 2. SUPERVISED PIPELINE (Regression / Classification)
        # -------------------------------------------------------------
        # Sanitize working data
        work_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
        if len(work_df) < 15:
            raise ValueError(f"Insufficient valid data ({len(work_df)} rows) to train machine learning models.")

        # Encode categorical features and target
        X_df = work_df[feature_cols].copy()
        y_series = work_df[target_col].copy()

        # Target encoding for classification
        label_encoder = None
        target_classes: List[str] = []
        if task_type == "classification":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_series.astype(str))
            target_classes = [str(c) for c in label_encoder.classes_]
            if len(target_classes) < 2:
                raise ValueError(f"Classification target '{target_col}' must have at least 2 distinct classes.")
        else:
            y = DataAnalyzer.safe_numeric_convert(y_series).to_numpy()

        # Preprocess features: One-hot encode categoricals, impute numerics
        X_encoded = pd.get_dummies(X_df, drop_first=True)
        # Fill any missing feature values with column medians
        X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))

        encoded_feature_names = list(X_encoded.columns)
        X = X_encoded.to_numpy()

        # Split train/test (leakage prevention)
        stratify = y if task_type == "classification" and len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Standard Scaler fitted ONLY on training split
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = cls._instantiate_model(algorithm, task_type, random_state)

        # Use scaled features for distance/gradient based models, unscaled for trees
        is_tree = any(t in algorithm.lower() for t in ["tree", "forest"])
        fit_X_train = X_train if is_tree else X_train_scaled
        fit_X_test = X_test if is_tree else X_test_scaled

        model.fit(fit_X_train, y_train)
        y_pred = model.predict(fit_X_test)

        # Feature importances or coefficients
        feature_importance = cls._extract_feature_importance(model, encoded_feature_names)

        # ----------------- REGRESSION EVALUATION -----------------
        if task_type == "regression":
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))

            # Sample comparison points for charting (top 40 points)
            sample_size = min(40, len(y_test))
            sample_actual = [round(float(v), 2) for v in y_test[:sample_size]]
            sample_pred = [round(float(v), 2) for v in y_pred[:sample_size]]

            explanation = (
                f"The {algorithm} achieved an R² Score of {r2:.3f} and MAE of {mae:,.2f}. "
                f"{'The model captures variance exceptionally well.' if r2 > 0.75 else ('Moderate predictive power; feature tuning recommended.' if r2 > 0.4 else 'Low explanatory power; additional features needed.')}"
            )

            return {
                "task_type": "regression",
                "algorithm": algorithm,
                "target_column": target_col,
                "feature_columns": feature_cols,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "metrics": {
                    "r2_score": round(r2, 4),
                    "mae": round(mae, 4),
                    "mse": round(mse, 4),
                    "rmse": round(rmse, 4),
                },
                "feature_importance": feature_importance,
                "actual_vs_predicted": {
                    "actual": sample_actual,
                    "predicted": sample_pred,
                },
                "explanation": explanation,
            }

        # ----------------- CLASSIFICATION EVALUATION -----------------
        else:
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            cm = confusion_matrix(y_test, y_pred).tolist()

            explanation = (
                f"The {algorithm} achieved an Accuracy of {acc*100:.1f}% and Weighted F1-Score of {f1:.3f}. "
                f"{'Model demonstrates robust generalization across classes.' if f1 > 0.80 else 'Model exhibits class imbalance or moderate misclassification.'}"
            )

            return {
                "task_type": "classification",
                "algorithm": algorithm,
                "target_column": target_col,
                "feature_columns": feature_cols,
                "classes": target_classes,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "metrics": {
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                },
                "confusion_matrix": {
                    "matrix": cm,
                    "labels": target_classes,
                },
                "feature_importance": feature_importance,
                "explanation": explanation,
            }

    @classmethod
    def _train_clustering(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_clusters: int = 3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Trains unsupervised K-Means Clustering on numerical feature subset."""
        numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_features) < 2:
            raise ValueError("K-Means Clustering requires at least 2 numeric features.")

        clean_data = df[numeric_features].dropna().copy()
        if len(clean_data) < 10:
            raise ValueError("Insufficient data rows for clustering analysis.")

        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(clean_data)

        k = max(2, min(n_clusters, len(clean_data) - 1))
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scaled_X)

        inertia = float(kmeans.inertia_)
        sil_score = float(silhouette_score(scaled_X, labels)) if len(np.unique(labels)) > 1 else 0.0

        cluster_counts = pd.Series(labels).value_counts().to_dict()
        cluster_summary = [
            {"cluster": int(cl), "count": int(cnt), "pct": round(cnt / len(labels) * 100, 1)}
            for cl, cnt in cluster_counts.items()
        ]

        # Cluster centroids back-transformed
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroid_dict = {}
        for idx, col in enumerate(numeric_features):
            centroid_dict[col] = [round(float(centroids[cl][idx]), 2) for cl in range(k)]

        explanation = (
            f"Segmented {len(clean_data):,} observations into {k} clusters with Silhouette Score of {sil_score:.3f}. "
            f"{'Strong cluster separation and cohesion.' if sil_score > 0.5 else 'Moderate cluster overlap; distinct groupings still discernible.'}"
        )

        return {
            "task_type": "clustering",
            "algorithm": "K-Means Clustering",
            "feature_columns": numeric_features,
            "n_clusters": k,
            "metrics": {
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil_score, 4),
            },
            "clusters": cluster_summary,
            "centroids": centroid_dict,
            "explanation": explanation,
        }

    @classmethod
    def compare_models(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> List[Dict[str, Any]]:
        """Trains all compatible algorithms for the task and benchmarks their scores."""
        task_type = cls.identify_task_type(df, target_col)
        algos = cls.ALGORITHMS.get(task_type, [])

        comparison: List[Dict[str, Any]] = []
        for algo in algos:
            try:
                res = cls.train_and_evaluate(
                    df, feature_cols, target_col, algorithm=algo, test_size=test_size, random_state=random_state
                )
                comparison.append(res)
            except Exception as e:
                comparison.append({"algorithm": algo, "error": str(e)})

        return comparison

    @classmethod
    def _instantiate_model(cls, algorithm: str, task_type: str, random_state: int):
        """Creates the appropriate scikit-learn estimator instance."""
        algo_l = algorithm.lower().replace("-", " ")
        if task_type == "regression":
            if "linear" in algo_l:
                return LinearRegression()
            elif "decision tree" in algo_l:
                return DecisionTreeRegressor(max_depth=6, random_state=random_state)
            elif "random forest" in algo_l:
                return RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state)
            return RandomForestRegressor(n_estimators=100, random_state=random_state)

        elif task_type == "classification":
            if "logistic" in algo_l:
                return LogisticRegression(max_iter=1000, random_state=random_state)
            elif "decision tree" in algo_l:
                return DecisionTreeClassifier(max_depth=6, random_state=random_state)
            elif "random forest" in algo_l:
                return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
            elif "neighbors" in algo_l or "knn" in algo_l:
                return KNeighborsClassifier(n_neighbors=5)
            return RandomForestClassifier(n_estimators=100, random_state=random_state)

        raise ValueError(f"Unknown algorithm: {algorithm}")

    @classmethod
    def _extract_feature_importance(cls, model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Extracts sorted normalized feature importances or model weights."""
        importances = []
        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
            total = sum(raw_imp) if sum(raw_imp) > 0 else 1.0
            for name, imp in zip(feature_names, raw_imp):
                importances.append({"feature": name, "importance": round(float(imp / total), 4)})
        elif hasattr(model, "coef_"):
            # Linear coefficients (take absolute magnitude)
            coefs = np.abs(model.coef_).flatten()
            total = sum(coefs) if sum(coefs) > 0 else 1.0
            for name, imp in zip(feature_names, coefs):
                importances.append({"feature": name, "importance": round(float(imp / total), 4)})

        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances[:10]  # Top 10 most influential features
