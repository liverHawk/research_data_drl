import pandas as pd
import imblearn.under_sampling as us
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
    df_resample = pd.concat([X, y], axis=1)
    return df_resample


def random(sampling_props: SamplingProps):
    rus = us.RandomUnderSampler(
        random_state=42,
    )
    return _base_sampling(rus, sampling_props), {
        "random_state": 42
    }


def cluster_centroid(sampling_props: SamplingProps):
    cc = us.ClusterCentroids(
        random_state=42,
    )
    return _base_sampling(cc, sampling_props), {
        "random_state": 42
    }


def edited_nearest_neighbors(sampling_props: SamplingProps):
    enn = us.EditedNearestNeighbours()
    return _base_sampling(enn, sampling_props), {}


def repeated_edited_nearest_neighbors(sampling_props: SamplingProps):
    renn = us.RepeatedEditedNearestNeighbours()
    return _base_sampling(renn, sampling_props), {}


def all_knn(sampling_props: SamplingProps):
    all_knn = us.AllKNN()
    return _base_sampling(all_knn, sampling_props), {}


def instance_hardness_threshold(sampling_props: SamplingProps):
    iht = us.InstanceHardnessThreshold()
    return _base_sampling(iht, sampling_props), {}


def near_miss(sampling_props: SamplingProps):
    nm = us.NearMiss()
    return _base_sampling(nm, sampling_props), {}


def neighbourhood_cleaning_rule(sampling_props: SamplingProps):
    ncr = us.NeighbourhoodCleaningRule()
    return _base_sampling(ncr, sampling_props), {}

def one_sided_selection(sampling_props: SamplingProps):
    oss = us.OneSidedSelection()
    return _base_sampling(oss, sampling_props), {}

def tomek_links(sampling_props: SamplingProps):
    tl = us.TomekLinks()
    return _base_sampling(tl, sampling_props), {}