from source.predict import predict_crop_and_risk


def test_prediction_output():
    result = predict_crop_and_risk(
        N=90,
        P=42,
        K=43,
        temperature=20.8,
        humidity=82.0,
        ph=6.5,
        rainfall=202.9
    )

    assert result is not None
    assert "recommended_crop" in result
    assert "risk_level" in result
    assert "advisory" in result