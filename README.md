# Email-Spam-Filtering

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset with an alternative encoding
data = pd.read_csv(r'C:\power bi\Email Spam Filtering\spam.csv', encoding='latin1')

# Check the first few rows of the dataset
print(data.head())

# Check the column names and their count
print(data.columns)
print(f"Number of columns: {len(data.columns)}")

# Assuming the columns are 'label' and 'text', if not, rename them accordingly
# Adjust the number of column names based on the actual number of columns
if len(data.columns) == 2:
    data.columns = ['label', 'text']
else:
    print("Unexpected number of columns. Please check the dataset.")
    print(data.columns)  # Print the actual column names for further inspection

# Preprocess the data if the columns are correctly named
if 'text' in data.columns and 'label' in data.columns:
    data['text'] = data['text'].str.lower()
    data['text'] = data['text'].str.replace('[^\w\s]', '', regex=True)

    # Ensure there are no missing values
    data.dropna(subset=['text', 'label'], inplace=True)

    # Extract features
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Add predictions to the dataset
    data['predicted_label'] = model.predict(vectorizer.transform(data['text']))

    # Output the data for Power BI
    output = data[['text', 'label', 'predicted_label']]
    print(output.head())
else:
    print("Required columns 'text' and 'label' are not present in the dataset.")
![Screenshot 2024-09-22 143008](https://github.com/user-attachments/assets/9b5cd1ae-892b-4ea9-9dcf-1420e1b87eff)
