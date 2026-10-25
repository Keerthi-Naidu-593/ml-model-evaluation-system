import numpy as np
import pandas as pd

from custom_dataset.evaluation import evaluate_dataset, profile_dataset


def _binary_dataset(rows=100):
    rng = np.random.RandomState(42)
    target = np.array(["No", "Yes"] * (rows // 2))
    values = rng.normal(size=rows) + (target == "Yes") * 0.9
    frame = pd.DataFrame(
        {
            "score": values,
            "segment": np.where(np.arange(rows) % 3, "retail", "business"),
            "target": target,
        }
    )
    frame.loc[0, "score"] = np.nan
    frame.loc[1, "segment"] = np.nan
    return frame


def test_custom_binary_evaluation_supports_categorical_values_and_missing_features():
    result = evaluate_dataset(_binary_dataset(), "target", positive_class_index=1)

    assert len(result["comparison"]) == 4
    assert len(result["cross_validation"]) == 4
    assert result["profile"].numeric_columns == ["score"]
    assert result["profile"].categorical_columns == ["segment"]
    assert result["comparison"]["ROC-AUC"].between(0, 1).all()


def test_custom_multiclass_evaluation_uses_macro_ovr_roc_auc():
    rng = np.random.RandomState(7)
    target = np.repeat(["alpha", "beta", "gamma"], 30)
    frame = pd.DataFrame(
        {
            "measurement": rng.normal(size=90) + np.repeat([0, 1, 2], 30),
            "category": np.tile(["north", "south", "east"], 30),
            "label": target,
        }
    )
    result = evaluate_dataset(frame, "label")

    assert "one-vs-rest macro" in result["evaluation_strategy"]
    assert len(result["comparison"]) == 4
    assert result["cross_validation"]["CV ROC-AUC"].between(0, 1).all()


def test_profile_preserves_uploaded_dataframe_and_identifies_class_distribution():
    frame = _binary_dataset()
    original = frame.copy(deep=True)
    profile = profile_dataset(frame, "target")

    pd.testing.assert_frame_equal(frame, original)
    assert profile.class_counts == {"No": 50, "Yes": 50}


def test_target_class_distribution_paginated_rendering():
    from streamlit.testing.v1 import AppTest

    test_code = """
import streamlit as st
import pandas as pd
from custom_dataset.app import _render_target_class_distribution
from custom_dataset.evaluation import profile_dataset

df = pd.DataFrame({
    'category': [f'cls_{i}' for i in range(12)],
    'feat': range(12)
})
profile = profile_dataset(df, 'category')
_render_target_class_distribution(profile, 'category')
"""
    at = AppTest.from_string(test_code, default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) == 5
    assert len(at.button) == 2
    assert at.button[0].disabled is True
    assert at.button[1].disabled is False

    # Click Next (1st click)
    at.button[1].click().run()
    assert not at.exception
    assert at.session_state["custom_dataset_class_dist_page"] == 2
    assert len(at.dataframe[0].value) == 5
    assert at.dataframe[0].value.iloc[0]["Class"] == "cls_5"

    # Click Next again (2nd click)
    at.button[1].click().run()
    assert not at.exception
    assert at.session_state["custom_dataset_class_dist_page"] == 3
    assert len(at.dataframe[0].value) == 2
    assert at.dataframe[0].value.iloc[0]["Class"] == "cls_10"
    assert at.button[1].disabled is True

    # Click Prev (3rd click)
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["custom_dataset_class_dist_page"] == 2
    assert at.dataframe[0].value.iloc[0]["Class"] == "cls_5"

    # Click Prev again (4th click)
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["custom_dataset_class_dist_page"] == 1
    assert at.button[0].disabled is True


def test_target_class_distribution_binary_has_no_pagination_controls():
    from streamlit.testing.v1 import AppTest

    test_code = """
import streamlit as st
import pandas as pd
from custom_dataset.app import _render_target_class_distribution
from custom_dataset.evaluation import profile_dataset

df = pd.DataFrame({
    'outcome': ['neg', 'pos'] * 10,
    'val': range(20)
})
profile = profile_dataset(df, 'outcome')
_render_target_class_distribution(profile, 'outcome')
"""
    at = AppTest.from_string(test_code, default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) == 2
    assert len(at.button) == 0

