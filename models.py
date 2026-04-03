"""
models.py
Financial Fraud Detection Models:
  1. Isolation Forest (Unsupervised)
  2. One-Class SVM (Unsupervised)
  3. Simple Autoencoder using sklearn (Unsupervised)
  4. Random Forest (Supervised - uses labels)
  5. Ensemble scoring
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor   # Used as Autoencoder approximation
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")


FEATURE_COLS = [
    "amount", "customer_age", "account_age_days", "credit_limit",
    "available_balance", "is_international", "hour_of_day",
    "day_of_week", "distance_from_home_km", "num_transactions_last_24h",
    "avg_transaction_amount_7d", "utilization_ratio", "amount_to_limit_ratio"
]

CAT_COLS = ["merchant_category"]


class FraudDetectionPipeline:
    """
    End-to-end fraud detection pipeline supporting multiple models.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.is_fitted = False
        self.feature_importances_ = None

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        for col in CAT_COLS:
            if col not in df.columns:
                df[col] = 0
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
                else:
                    df[col] = 0
        return df

    def _prepare_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        df = self._encode_categoricals(df, fit=fit)
        all_cols = FEATURE_COLS + CAT_COLS
        available = [c for c in all_cols if c in df.columns]
        X = df[available].fillna(0).values

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X

    # ------------------------------------------------------------------
    # Model Training
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame):
        """Train all models on the provided DataFrame."""
        print("Preparing features...")
        X = self._prepare_features(df, fit=True)
        y = df["is_fraud"].values if "is_fraud" in df.columns else None

        # --- Isolation Forest ---
        print("Training Isolation Forest...")
        self.models["isolation_forest"] = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            max_samples="auto",
            random_state=42,
            n_jobs=-1
        )
        self.models["isolation_forest"].fit(X)

        # --- One-Class SVM (train on normal transactions only) ---
        print("Training One-Class SVM...")
        if y is not None:
            X_normal = X[y == 0][:3000]  # Use subset for speed
        else:
            X_normal = X[:3000]
        self.models["ocsvm"] = OneClassSVM(
            kernel="rbf",
            nu=0.05,
            gamma="scale"
        )
        self.models["ocsvm"].fit(X_normal)

        # --- Autoencoder (MLP-based reconstruction) ---
        print("Training Autoencoder...")
        n_features = X.shape[1]
        self.models["autoencoder"] = MLPRegressor(
            hidden_layer_sizes=(32, 8, 32),
            activation="relu",
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        X_normal_full = X[y == 0] if y is not None else X
        self.models["autoencoder"].fit(X_normal_full, X_normal_full)

        # --- Random Forest (Supervised) ---
        if y is not None:
            print("Training Random Forest (Supervised)...")
            self.models["random_forest"] = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
            self.models["random_forest"].fit(X, y)
            self.feature_importances_ = self.models["random_forest"].feature_importances_

            print("Training Gradient Boosting...")
            self.models["gradient_boosting"] = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
            self.models["gradient_boosting"].fit(X, y)

        self.is_fitted = True
        print("All models trained successfully!")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _anomaly_score_iso(self, X: np.ndarray) -> np.ndarray:
        """Isolation Forest: returns 0-1 probability (higher = more anomalous)."""
        raw = self.models["isolation_forest"].score_samples(X)
        # Normalize to 0-1 (invert: lower score = more anomalous)
        score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return score

    def _anomaly_score_ocsvm(self, X: np.ndarray) -> np.ndarray:
        """One-Class SVM: returns 0-1 probability."""
        raw = self.models["ocsvm"].score_samples(X)
        score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return score

    def _anomaly_score_autoencoder(self, X: np.ndarray) -> np.ndarray:
        """Autoencoder: reconstruction error as anomaly score."""
        X_reconstructed = self.models["autoencoder"].predict(X)
        mse = np.mean((X - X_reconstructed) ** 2, axis=1)
        score = (mse - mse.min()) / (mse.max() - mse.min() + 1e-9)
        return score

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Run all models and return enriched DataFrame with scores."""
        X = self._prepare_features(df, fit=False)
        results = df.copy()

        # Unsupervised scores
        results["score_isolation_forest"] = self._anomaly_score_iso(X)
        results["score_ocsvm"] = self._anomaly_score_ocsvm(X)
        results["score_autoencoder"] = self._anomaly_score_autoencoder(X)

        # Supervised scores
        if "random_forest" in self.models:
            results["score_random_forest"] = self.models["random_forest"].predict_proba(X)[:, 1]
        if "gradient_boosting" in self.models:
            results["score_gradient_boosting"] = self.models["gradient_boosting"].predict_proba(X)[:, 1]

        # Ensemble: weighted average
        score_cols = [c for c in results.columns if c.startswith("score_")]
        weights = {
            "score_isolation_forest": 0.15,
            "score_ocsvm": 0.10,
            "score_autoencoder": 0.10,
            "score_random_forest": 0.35,
            "score_gradient_boosting": 0.30,
        }
        total_weight = sum(weights[c] for c in score_cols if c in weights)
        ensemble = sum(
            results[c] * weights.get(c, 0.2) for c in score_cols
        ) / total_weight
        results["ensemble_score"] = ensemble
        results["is_fraud_predicted"] = (ensemble >= threshold).astype(int)

        # Risk level
        results["risk_level"] = pd.cut(
            ensemble,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["Low", "Medium", "High", "Critical"]
        )

        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, threshold: float = 0.5) -> dict:
        """Return comprehensive evaluation metrics."""
        results = self.predict(df, threshold)
        y_true = df["is_fraud"].values
        metrics = {}

        for col in [c for c in results.columns if c.startswith("score_")] + ["ensemble_score"]:
            model_name = col.replace("score_", "").replace("_score", "ensemble")
            y_score = results[col].values
            y_pred = (y_score >= threshold).astype(int)
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = 0.0
            metrics[model_name] = {
                "auc_roc": round(auc, 4),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "ap": round(average_precision_score(y_true, y_score), 4),
            }

        metrics["confusion_matrix"] = confusion_matrix(
            y_true, results["is_fraud_predicted"].values
        ).tolist()
        metrics["classification_report"] = classification_report(
            y_true, results["is_fraud_predicted"].values, output_dict=True
        )

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = "models/fraud_model.joblib"):
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path: str = "models/fraud_model.joblib") -> "FraudDetectionPipeline":
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model

    def predict_single(self, transaction: dict, threshold: float = 0.5) -> dict:
        """Predict fraud probability for a single transaction dict."""
        df = pd.DataFrame([transaction])
        # Add derived features if missing
        if "utilization_ratio" not in df.columns:
            df["utilization_ratio"] = (
                (df.get("credit_limit", 50000) - df.get("available_balance", 25000))
                / df.get("credit_limit", 50000)
            )
        if "amount_to_limit_ratio" not in df.columns:
            df["amount_to_limit_ratio"] = df.get("amount", 100) / df.get("credit_limit", 50000)

        result = self.predict(df, threshold)
        row = result.iloc[0]
        return {
            "ensemble_score": round(float(row["ensemble_score"]), 4),
            "risk_level": str(row["risk_level"]),
            "is_fraud_predicted": int(row["is_fraud_predicted"]),
            "score_isolation_forest": round(float(row["score_isolation_forest"]), 4),
            "score_ocsvm": round(float(row["score_ocsvm"]), 4),
            "score_autoencoder": round(float(row["score_autoencoder"]), 4),
            "score_random_forest": round(float(row.get("score_random_forest", 0)), 4),
            "score_gradient_boosting": round(float(row.get("score_gradient_boosting", 0)), 4),
        }


if __name__ == "__main__":
    from data_generator import generate_transactions
    df = generate_transactions(5000)
    pipeline = FraudDetectionPipeline()
    pipeline.train(df)
    metrics = pipeline.evaluate(df)
    print("\n=== Evaluation Results ===")
    for model, m in metrics.items():
        if isinstance(m, dict):
            print(f"\n{model}: AUC={m.get('auc_roc')} F1={m.get('f1')}")
