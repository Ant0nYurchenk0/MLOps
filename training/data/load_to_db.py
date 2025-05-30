from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
import os

load_dotenv()

# DB config
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Paths
TRAIN_PATH = os.getenv("TRAIN_PATH")
TEST_PATH = os.getenv("TEST_PATH")
HOLDOUT_PATH = os.getenv("HOLDOUT_PATH")

# Create DB connection
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
# Load and prepare training data
df_train_full = pd.read_csv(TRAIN_PATH).fillna("")
df_holdout = df_train_full.sample(n=100_000, random_state=42)
df_train = df_train_full.drop(df_holdout.index)

df_train["prediction"] = None
df_train["ready_to_use"] = True


# Write main training data to DB
df_train.to_sql("questions", con=engine, index=False, if_exists="replace")
print(f"✅ Loaded {len(df_train)} rows into table 'questions'")

# Load and write test set to DB
df_test = pd.read_csv(TEST_PATH).fillna("")

df_test.to_sql("questions_test", con=engine, index=False, if_exists="replace")
print(f"✅ Loaded {len(df_test)} rows into table 'questions_test'")

# Save holdout chunk to CSV
df_holdout.to_csv(HOLDOUT_PATH, index=False)
print(f"📦 Saved {len(df_holdout)} rows to {HOLDOUT_PATH} for simulated future data.")
