import pandas as pd
import imblearn.over_sampling as os
from dataclasses import dataclass


@dataclass
class SamplingProps:
    df: pd.DataFrame
    y_column_name: str


def _base_sampling(func, sampling_props: SamplingProps):
    X, y = func.fit_resample(
        sampling_props.df.drop(columns=[sampling_props.y_column_name]),
        sampling_props.df[sampling_props.y_column_name]
    )
    df_resample = pd.concat([X, y])
    return df_resample


def random(sampling_props: SamplingProps):
    ros = os.RandomOverSampler(
        random_state=42,
    )
    return _base_sampling(ros, sampling_props), {
        "random_state": 42
    }
    

def smote(sampling_props: SamplingProps):
    min_samples_num = sampling_props.df[sampling_props.y_column_name].value_counts().min()

    smote = os.SMOTE(
        random_state=42,
        k_neighbors=max(2, min_samples_num - 1),
    )
    return _base_sampling(smote, sampling_props), {
        "random_state": 42,
        "k_neighbors": max(2, min_samples_num - 1)
    }


def adasyn(sampling_props: SamplingProps):
    min_samples_num = sampling_props.df[sampling_props.y_column_name].value_counts().min()

    adasyn = os.ADASYN(
        random_state=42,
        n_neighbors=max(2, min_samples_num - 1),
    )
    return _base_sampling(adasyn, sampling_props), {
        "random_state": 42,
        "n_neighbors": max(2, min_samples_num - 1)
    }


def borderline_smote(sampling_props: SamplingProps):
    min_samples_num = sampling_props.df[sampling_props.y_column_name].value_counts().min()

    borderline_smote = os.BorderlineSMOTE(
        random_state=42,
        k_neighbors=max(2, min_samples_num - 1),
        m_neighbors=max(2, min_samples_num - 1),
    )
    return _base_sampling(borderline_smote, sampling_props), {
        "random_state": 42,
        "k_neighbors": max(2, min_samples_num - 1),
        "m_neighbors": max(2, min_samples_num - 1)
    }


def kmeans_smote(sampling_props: SamplingProps):
    min_samples_num = sampling_props.df[sampling_props.y_column_name].value_counts().min()

    kmeans_smote = os.KMeansSMOTE(
        random_state=42,
        k_neighbors=max(2, min_samples_num - 1),
    )
    return _base_sampling(kmeans_smote, sampling_props), {
        "random_state": 42,
        "k_neighbors": max(2, min_samples_num - 1)
    }


def svmsmote(sampling_props: SamplingProps):
    min_samples_num = sampling_props.df[sampling_props.y_column_name].value_counts().min()

    svmsmote = os.SVMSMOTE(
        random_state=42,
        k_neighbors=max(2, min_samples_num - 1),
    )
    return _base_sampling(svmsmote, sampling_props), {
        "random_state": 42,
        "k_neighbors": max(2, min_samples_num - 1)
    }