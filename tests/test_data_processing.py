from source.data_processing import load_and_process_data


def test_load_and_process_data():
    df = load_and_process_data("data/crop_recommendation.csv")

    assert df is not None
    assert "risk_level" in df.columns
    assert "advisory" in df.columns
    assert len(df) > 0