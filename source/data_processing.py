import pandas as pd


def create_risk_level(row):
    risk_score = 0

    # Rainfall risk
    if row['rainfall'] < 40 or row['rainfall'] > 250:
        risk_score += 2
    elif row['rainfall'] < 60 or row['rainfall'] > 220:
        risk_score += 1

    # pH risk
    if row['ph'] < 5.5 or row['ph'] > 8.0:
        risk_score += 2
    elif row['ph'] < 6.0 or row['ph'] > 7.5:
        risk_score += 1

    # Nutrient risk
    npk_avg = (row['N'] + row['P'] + row['K']) / 3
    if npk_avg < 30:
        risk_score += 2
    elif npk_avg < 50:
        risk_score += 1

    if risk_score >= 5:
        return "High Risk"
    elif risk_score >= 3:
        return "Moderate Risk"
    else:
        return "Safe"


def create_advisory(row):
    if row['rainfall'] > 200:
        return "Suitable for high rainfall crops. Ensure proper drainage."
    elif row['rainfall'] < 50:
        return "Low rainfall detected. Irrigation is recommended."
    elif row['ph'] < 5.8:
        return "Soil is acidic. Consider pH balancing before cultivation."
    elif row['ph'] > 7.8:
        return "Soil is alkaline. Soil treatment may improve productivity."
    else:
        return "Conditions are generally suitable for cultivation."


def load_and_process_data(path):
    df = pd.read_csv(path)

    df['risk_level'] = df.apply(create_risk_level, axis=1)
    df['advisory'] = df.apply(create_advisory, axis=1)

    return df