SELECTION_RECORDS = [
    {
        "fold": 1,
        "metal": "Sn",
        "max_features": 10,
        "selected_count": 10,
        "wavelengths": [429.95, 484.30, 901.78, 912.15, 916.30, 1059.12, 1326.97, 1454.44, 1478.41, 1799.44],
    },
    {
        "fold": 1,
        "metal": "Bi",
        "max_features": 13,
        "selected_count": 13,
        "wavelengths": [429.95, 484.30, 665.85, 901.78, 905.93, 912.15, 916.30, 1059.12, 1326.97, 1454.44, 1478.41, 1695.59, 1799.44],
    },
    {
        "fold": 2,
        "metal": "Sn",
        "max_features": 10,
        "selected_count": 5,
        "wavelengths": [905.93, 908.01, 926.64, 928.70, 1493.10],
    },
    {
        "fold": 2,
        "metal": "Bi",
        "max_features": 13,
        "selected_count": 5,
        "wavelengths": [905.93, 908.01, 926.64, 928.70, 1493.10],
    },
    {
        "fold": 3,
        "metal": "Sn",
        "max_features": 10,
        "selected_count": 10,
        "wavelengths": [429.95, 702.71, 910.08, 918.37, 936.96, 1326.97, 1478.41, 1491.26, 1500.42, 1799.44],
    },
    {
        "fold": 3,
        "metal": "Bi",
        "max_features": 13,
        "selected_count": 13,
        "wavelengths": [429.95, 702.71, 910.08, 916.30, 918.37, 936.96, 1326.97, 1368.59, 1478.41, 1491.26, 1500.42, 1565.83, 1799.44],
    },
    {
        "fold": 4,
        "metal": "Sn",
        "max_features": 10,
        "selected_count": 10,
        "wavelengths": [473.02, 540.33, 663.68, 912.15, 916.30, 924.57, 932.83, 1059.12, 1232.90, 1326.97],
    },
    {
        "fold": 4,
        "metal": "Bi",
        "max_features": 13,
        "selected_count": 13,
        "wavelengths": [473.02, 511.27, 540.33, 663.68, 912.15, 916.30, 924.57, 932.83, 1059.12, 1079.19, 1232.90, 1326.97, 1799.44],
    },
    {
        "fold": 5,
        "metal": "Sn",
        "max_features": 10,
        "selected_count": 10,
        "wavelengths": [427.67, 436.77, 813.81, 901.78, 910.08, 939.02, 1328.87, 1422.90, 1489.43, 1799.44],
    },
    {
        "fold": 5,
        "metal": "Bi",
        "max_features": 13,
        "selected_count": 13,
        "wavelengths": [427.67, 436.77, 813.81, 901.78, 910.08, 926.64, 939.02, 1059.12, 1328.87, 1422.90, 1489.43, 1756.47, 1799.44],
    },
]


def folds() -> list[int]:
    return sorted({int(record["fold"]) for record in SELECTION_RECORDS})


def selected_wavelengths(fold: int, metal: str) -> list[float]:
    for record in SELECTION_RECORDS:
        if int(record["fold"]) == fold and record["metal"] == metal:
            return list(record["wavelengths"])
    raise ValueError(f"No selection record found for fold={fold}, metal={metal}.")
