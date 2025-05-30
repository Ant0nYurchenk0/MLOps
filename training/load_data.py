from sqlalchemy import create_engine
import pandas as pd


def load_data(db_uri: str):
    """
    Loads training and test sets from PostgreSQL.

    Returns:
        train_texts: List[str] of training question_text
        train_labels: np.ndarray of training targets
        test_texts: List[str] of test question_text
    """
    engine = create_engine(db_uri)

    # Training data
    train_query = """
    SELECT question_text, target
    FROM questions
    WHERE ready_to_use = TRUE
    """
    train_df = pd.read_sql(train_query, con=engine)

    # Test data
    test_query = """
    SELECT question_text
    FROM questions_test
    """
    test_df = pd.read_sql(test_query, con=engine)

    train_texts = train_df["question_text"].tolist()
    train_labels = train_df["target"].values
    test_texts = test_df["question_text"].tolist()

    return train_texts, train_labels, test_texts
