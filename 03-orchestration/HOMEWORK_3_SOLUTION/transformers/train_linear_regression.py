import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train_linear_regression(df: pd.DataFrame, **kwargs) -> tuple:
    # Select columns for one-hot encoding
    cols = ['PULocationID', 'DOLocationID']
    df[cols] = df[cols].astype(str)

    # Convert DataFrame to list of dictionaries
    records = df[cols].to_dict('records')

    # Fit DictVectorizer
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit(records)

    # Transform records to feature matrix
    feature_matrix = vectorizer.transform(records)

    # Prepare target variable (duration in minutes)
    y = df['duration'].values

    # Train linear regression model
    lr_model = LinearRegression()
    lr_model.fit(feature_matrix, y)

    # Print the intercept for reference
    print(f"Model intercept: {lr_model.intercept_}")

    return vectorizer, lr_model


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
