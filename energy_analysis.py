import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("campus_energy.csv")

print("\n===== RAW ENERGY DATA =====\n")
print(data)

# Convert month to numeric for trend analysis
month_map = {"Jan": 1, "Feb": 2, "Mar": 3}
data["Month_Num"] = data["Month"].map(month_map)

# Average usage
avg_usage = data["Units_Consumed"].mean()

# High usage detection
data["Usage_Status"] = data["Units_Consumed"].apply(
    lambda x: "High Usage" if x > avg_usage else "Normal Usage"
)

# Anomaly detection (very high usage)
anomaly_threshold = avg_usage * 1.25
data["Anomaly"] = data["Units_Consumed"].apply(
    lambda x: "Yes" if x > anomaly_threshold else "No"
)

# AI Insight generation
def generate_insight(row):
    if row["Anomaly"] == "Yes":
        return "Unusual spike detected. Immediate energy audit recommended."
    elif row["Usage_Status"] == "High Usage":
        return "High energy usage. Suggest optimization and awareness measures."
    else:
        return "Energy usage is stable and within limits."

data["AI_Insight"] = data.apply(generate_insight, axis=1)

print("\n===== AI ENERGY INSIGHTS =====\n")
print(data[["Building", "Month", "Units_Consumed", "Usage_Status", "Anomaly", "AI_Insight"]])

# Ranking buildings by energy usage
ranking = data.groupby("Building")["Units_Consumed"].mean().sort_values(ascending=False)
print("\n===== BUILDING ENERGY RANKING =====\n")
print(ranking)

# Simple future prediction (trend-based)
future_prediction = data.groupby("Building")["Units_Consumed"].mean() * 1.05
print("\n===== PREDICTED NEXT MONTH USAGE (ESTIMATE) =====\n")
print(future_prediction)

# Visualization
plt.figure(figsize=(9,5))
plt.bar(ranking.index, ranking.values)
plt.title("Average Energy Consumption by Building")
plt.xlabel("Building")
plt.ylabel("Units Consumed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
