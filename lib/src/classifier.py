import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


NORMALIZE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "SYN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min"
]


def check_numeric_features(X: pd.DataFrame):
    """
    Check if all features in the DataFrame are numeric.
    Raises ValueError if any feature is not numeric.
    """
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise ValueError(
                f"Feature '{col}' is not numeric.",
                "Convert it to numeric before fitting."
            )


class ImprovedC45:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        load_path=None
    ):
        if load_path:
            self.clf = joblib.load(load_path)
        else:
            self.clf = DecisionTreeClassifier(
                criterion='entropy',
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
        self.rolling_mean = None
        self.rolling_std = None
    
    def _rolling_normalize(self, X: pd.DataFrame):
        X_norm = X.copy()
        rolling = X_norm[NORMALIZE_COLUMNS].rolling(window=5, min_periods=1)
        self.rolling_mean = rolling.mean()
        self.rolling_std = rolling.std()
        X_norm[NORMALIZE_COLUMNS] = (X_norm[NORMALIZE_COLUMNS] - self.rolling_mean) / self.rolling_std
        X_norm[NORMALIZE_COLUMNS].fillna(0, inplace=True)
        return X_norm

    def fit(self, X: pd.DataFrame, y):
        check_numeric_features(X)
        X = self._rolling_normalize(X)
        self.clf.fit(X, y)

    def predict(self, X):
        X = self._rolling_normalize(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = self._rolling_normalize(X)
        return self.clf.predict_proba(X)
    
    def save(self, path):
        joblib.dump(self.clf, path)

