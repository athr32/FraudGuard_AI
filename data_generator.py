"""
data_generator.py
Generates synthetic financial transaction data for fraud detection.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_transactions(n_transactions: int = 10000, fraud_ratio: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic transaction data mimicking real-world banking data.

    Features:
        - transaction_id, timestamp, amount, merchant_category
        - customer_age, account_age_days, credit_limit, available_balance
        - transaction_country, is_international
        - hour_of_day, day_of_week
        - distance_from_home_km
        - is_fraud (label)
    """
    np.random.seed(seed)
    random.seed(seed)

    n_fraud = int(n_transactions * fraud_ratio)
    n_normal = n_transactions - n_fraud

    merchant_categories = [
        "grocery", "retail", "restaurant", "gas_station", "online",
        "travel", "healthcare", "entertainment", "atm", "utilities"
    ]
    countries = ["India", "USA", "UK", "Germany", "UAE", "Singapore", "China", "Australia"]

    def gen_normal():
        records = []
        base_date = datetime(2024, 1, 1)
        for i in range(n_normal):
            ts = base_date + timedelta(
                days=random.randint(0, 364),
                hours=random.randint(6, 22),
                minutes=random.randint(0, 59)
            )
            amount = np.random.lognormal(mean=3.5, sigma=1.0)
            amount = np.clip(amount, 1, 5000)
            credit_limit = np.random.choice([10000, 25000, 50000, 100000, 200000])
            records.append({
                "transaction_id": f"TXN{i:07d}",
                "timestamp": ts,
                "amount": round(amount, 2),
                "merchant_category": random.choice(merchant_categories[:7]),
                "customer_age": np.random.randint(22, 70),
                "account_age_days": np.random.randint(180, 3650),
                "credit_limit": credit_limit,
                "available_balance": round(credit_limit * np.random.uniform(0.3, 0.95), 2),
                "transaction_country": "India",
                "is_international": 0,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "distance_from_home_km": round(np.random.exponential(scale=10), 2),
                "num_transactions_last_24h": np.random.randint(0, 8),
                "avg_transaction_amount_7d": round(amount * np.random.uniform(0.7, 1.3), 2),
                "is_fraud": 0
            })
        return records

    def gen_fraud():
        records = []
        base_date = datetime(2024, 1, 1)
        for i in range(n_fraud):
            ts = base_date + timedelta(
                days=random.randint(0, 364),
                hours=random.choice([0, 1, 2, 3, 4, 23]),   # odd hours
                minutes=random.randint(0, 59)
            )
            amount = np.random.choice([
                np.random.uniform(3000, 50000),   # large amounts
                np.random.uniform(1, 5),           # micro-transactions
            ])
            credit_limit = np.random.choice([10000, 25000, 50000, 100000, 200000])
            is_intl = np.random.choice([0, 1], p=[0.3, 0.7])
            records.append({
                "transaction_id": f"TXN_F{i:06d}",
                "timestamp": ts,
                "amount": round(amount, 2),
                "merchant_category": random.choice(merchant_categories),
                "customer_age": np.random.randint(18, 80),
                "account_age_days": np.random.randint(1, 90),   # new accounts
                "credit_limit": credit_limit,
                "available_balance": round(credit_limit * np.random.uniform(0.01, 0.15), 2),  # low balance
                "transaction_country": random.choice(countries[1:]) if is_intl else "India",
                "is_international": is_intl,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "distance_from_home_km": round(np.random.uniform(500, 5000), 2),  # far from home
                "num_transactions_last_24h": np.random.randint(10, 40),  # burst activity
                "avg_transaction_amount_7d": round(np.random.uniform(100, 500), 2),
                "is_fraud": 1
            })
        return records

    normal_records = gen_normal()
    fraud_records = gen_fraud()

    df = pd.DataFrame(normal_records + fraud_records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["utilization_ratio"] = (df["credit_limit"] - df["available_balance"]) / df["credit_limit"]
    df["amount_to_limit_ratio"] = df["amount"] / df["credit_limit"]

    return df


def generate_streaming_transaction(fraud_prob: float = 0.05) -> dict:
    """Simulate a single streaming transaction for real-time monitoring."""
    np.random.seed(None)
    is_fraud = np.random.random() < fraud_prob
    now = datetime.now()

    if is_fraud:
        amount = float(np.random.choice([
            np.random.uniform(3000, 20000),
            np.random.uniform(1, 5)
        ]))
        hour = random.choice([0, 1, 2, 3, 23])
        distance = float(np.random.uniform(500, 5000))
        n_tx = random.randint(10, 30)
        country = random.choice(["USA", "UK", "UAE", "China"])
        is_intl = 1
        balance_ratio = float(np.random.uniform(0.01, 0.15))
    else:
        amount = float(np.clip(np.random.lognormal(3.5, 1.0), 1, 5000))
        hour = random.randint(7, 22)
        distance = float(np.random.exponential(10))
        n_tx = random.randint(0, 8)
        country = "India"
        is_intl = 0
        balance_ratio = float(np.random.uniform(0.3, 0.95))

    credit_limit = float(random.choice([10000, 25000, 50000, 100000]))
    return {
        "transaction_id": f"LIVE_{now.strftime('%H%M%S%f')[:10]}",
        "timestamp": now.isoformat(),
        "amount": round(amount, 2),
        "merchant_category": random.choice(["grocery", "online", "travel", "atm", "retail"]),
        "customer_age": random.randint(20, 70),
        "account_age_days": random.randint(1, 3650),
        "credit_limit": credit_limit,
        "available_balance": round(credit_limit * balance_ratio, 2),
        "transaction_country": country,
        "is_international": is_intl,
        "hour_of_day": hour,
        "day_of_week": now.weekday(),
        "distance_from_home_km": round(distance, 2),
        "num_transactions_last_24h": n_tx,
        "avg_transaction_amount_7d": round(float(np.random.uniform(100, 2000)), 2),
        "is_fraud_actual": int(is_fraud)
    }


if __name__ == "__main__":
    df = generate_transactions(10000)
    print(df.head())
    print(f"\nTotal: {len(df)} | Fraud: {df['is_fraud'].sum()} | Normal: {(df['is_fraud'] == 0).sum()}")
    df.to_csv("data/transactions.csv", index=False)
    print("Saved to data/transactions.csv")
