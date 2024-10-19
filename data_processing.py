import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# make sure necessary NLTK data is available
nltk.download('stopwords')

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Clean the comments using the stopwords
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '',text) # Remove non-alphabetic characters
    text = text.lower() # set it all to lowercase
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')] # Remove stopwords
    return ' '.join(tokens)

# Preprocess the data
def preprocess_data(data):
    data['clean_comments'] = data['PNT_ATRISKNOTES_TX'].apply(clean_text)
    return data

# Split the data
def split_data(data,test_size=0.2):
    X_train, X_test = train_test_split(data, test_size=test_size, random_state=42)
    return X_train, X_test

if __name__ == "__main__":
    data = load_data("CORE_HackOhio_subset_cleaned_downsampled 1.csv")
    cleaned_data = preprocess_data(data)
    X_train, X_test = split_data(cleaned_data)
    X_train.to_csv('train_data.csv',index = False)
    X_test.to_csv('test_data.csv',index = False)

