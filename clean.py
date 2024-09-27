import pandas as pd
import re
from langdetect import detect
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class DataCleaner:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None

    def load_data(self):
        # Load the CSV file
        self.data = pd.read_csv(self.input_file)
        print("Number of rows before filtering:", len(self.data))

    def clean_text(self):
        # Remove special characters
        self.data['comment'] = self.data['comment'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

        # Strip leading/trailing whitespace and replace multiple spaces with a single space
        self.data['comment'] = self.data['comment'].str.strip()
        self.data['comment'] = self.data['comment'].str.replace(r'\s+', ' ', regex=True)

    def filter_english_comments(self):
        # Define a function to filter only English comments
        def is_english(text):
            try:
                return detect(text) == 'en'
            except:
                return False

        # Filter rows where comments are detected as English
        self.data = self.data[self.data['comment'].apply(is_english)]

    def remove_links(self):
        # Remove rows that contain URLs
        self.data = self.data[~self.data['comment'].str.contains(r'http\S+|www\S+', case=False, na=False)]

    def remove_stopwords(self):
        # Remove stopwords from comments
        self.data['comment'] = self.data['comment'].apply(
            lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
        )

    def filter_by_length(self):
        # Filter rows where the comment has at least 50 characters
        self.data = self.data[self.data['comment'].str.len() >= 50]

    def remove_duplicates(self):
        # Remove duplicate rows based on the 'comment' column
        self.data = self.data.drop_duplicates(subset=['comment'])

    def save_data(self):
        # Save the cleaned data to a new CSV file
        self.data.to_csv(self.output_file, index=False)
        print("Number of rows after filtering:", len(self.data))

    def run(self):
        self.load_data()
        self.remove_duplicates()
        self.clean_text()
        self.filter_english_comments()
        self.remove_links()
        self.filter_by_length()
        self.remove_stopwords()
        self.save_data()

# Example usage:
cleaner = DataCleaner(input_file='scraped_threads.csv', output_file='./data/filtered_threads_file.csv')
cleaner.run()
