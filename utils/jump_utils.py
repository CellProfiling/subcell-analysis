import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pycytominer import normalize, feature_select
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


def match_matrix(df, label_col):
    binarizer = MultiLabelBinarizer()
    if not isinstance(df[label_col][0], list):
        labels = [[l] for l in df[label_col]]
    else:
        labels = df[label_col].to_list()
    one_hots = binarizer.fit_transform(labels)
    match_matrix = (one_hots @ one_hots.T).astype(bool)
    match_matrix = match_matrix > 0
    match_matrix[np.diag_indices_from(match_matrix)] = 0
    return match_matrix


def sim_matrix(df, feature_cols):
    features = df[feature_cols].to_numpy()
    sim_matrix = cosine_similarity(features)
    sim_matrix = sim_matrix.clip(-1, 1)
    sim_matrix[np.diag_indices_from(sim_matrix)] = -1
    return sim_matrix


def nn_accuracy(match_matrix, sim_matrix, plate_block=None):
    assert match_matrix.shape == sim_matrix.shape
    if plate_block is None:
        plate_block = np.ones(sim_matrix.shape).astype(bool)
    assert match_matrix.shape == plate_block.shape

    # Block same plate
    match_matrix_blocked = match_matrix * plate_block
    sim_matrix_blocked = sim_matrix.copy()
    sim_matrix_blocked[~plate_block] = -1

    # Find idxs with highest similarity
    row_max = np.argmax(sim_matrix_blocked, axis=0)
    idxs = np.stack((row_max, np.arange(len(row_max))))
    i = np.ravel_multi_index(idxs, sim_matrix_blocked.shape)

    # Check if idxs are matches
    matches = match_matrix_blocked.take(i)
    accuracy = np.mean(matches)

    return accuracy


def normalize_step(df, metadata_cols, feature_cols, method):
    if method in ["mad_robustize", "standardize"]:
        groups = df.groupby("plate")
        normalized_plates = []
        for _, plate_df in groups:
            normalized_plate = normalize(
                profiles=plate_df,
                features=feature_cols,
                meta_features=metadata_cols,
                method=method,
                mad_robustize_epsilon=1e-18,  # Maybe try to change
            )
            normalized_plates.append(normalized_plate)

        res_df = pd.concat(normalized_plates)
    elif method in ["ZCA", "ZCA-cor", "PCA", "PCA-cor"]:
        # SPHERIZE to BATCH CORRECT
        res_df = normalize(
            profiles=df,
            features=feature_cols,
            meta_features=metadata_cols,
            samples="compound == 'DMSO'",
            method="spherize",
            spherize_method=method,
            spherize_epsilon=1e-06,
        )
        feature_cols = res_df.columns[~res_df.columns.isin(metadata_cols)].tolist()
    else:
        res_df = df.copy(deep=True)
    return res_df, feature_cols


def elim_corr(df, metadata_cols, feature_cols):
    res_df = feature_select(
        profiles=df,
        features=feature_cols,
        samples="all",
        image_features=False,
        corr_threshold=0.9,
        corr_method="pearson",
        freq_cut=0.05,
        unique_cut=0.01,
        operation=["variance_threshold", "correlation_threshold"],
    )
    feature_cols = res_df.columns[~res_df.columns.isin(metadata_cols)].tolist()
    return res_df, feature_cols


# Source: https://github.com/broadinstitute/DeepProfilerExperiments/blob/master/profiling/metrics.py
def interpolated_precision_recall_curve(Y_true_matrix, Y_predict_matrix):
    """Compute the average precision / recall curve over all queries in a matrix.
    That is, consider each point in the graph as a query and evaluate all nearest neighbors until
    all positives have been found. Y_true_matrix is a binary matrix, Y_predict_matrix is a
    continuous matrix. Each row in the matrices is a query, and for each one PR curve can be computed.
    Since each PR curve has a different number of recall points, the curves are interpolated to
    cover the max number of recall points. This is standard practice in Information Retrieval research
    """

    from sklearn.metrics import precision_recall_curve

    # Suppress self matching
    Y_predict_matrix[np.diag_indices(Y_predict_matrix.shape[0])] = (
        -1
    )  # Assuming Pearson correlation as the metric
    Y_true_matrix[np.diag_indices(Y_true_matrix.shape[0])] = (
        False  # Assuming a binary matrix
    )

    # Prepare axes
    recall_axis = np.linspace(0.0, 1.0, num=Y_true_matrix.shape[0])[::-1]
    precision_axis = []

    # Each row in the matrix is one query
    is_query = Y_true_matrix.sum(axis=0) > 1
    for t in range(Y_true_matrix.shape[0]):
        if not is_query[t]:
            continue
        # Compute precision / recall for each query
        precision_t, recall_t, _ = precision_recall_curve(
            Y_true_matrix[t, :], Y_predict_matrix[t, :]
        )

        # Interpolate max precision at all recall points
        max_precision = np.maximum.accumulate(precision_t)
        interpolated_precision = np.zeros_like(recall_axis)

        j = 0
        for i in range(recall_axis.shape[0]):
            interpolated_precision[i] = max_precision[j]
            while recall_axis[i] < recall_t[j]:
                j += 1

        # Store interpolated results for query
        precision_axis.append(interpolated_precision[:, np.newaxis])

    return recall_axis, np.mean(np.concatenate(precision_axis, axis=1), axis=1)
