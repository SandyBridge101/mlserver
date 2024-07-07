import numpy as np 
import pandas as pd
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

sheet_name='products_dataset'
sheet_id='1izclQ3BwjiqxbqycRj0EF9NxLd63pDOEdJg6Wt4n6T8'
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"



def get_list(entry):
    df_Ama = pd.read_csv(url)

    # Function to get the POS tag for accurate lemmatization
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    # Function to concatenate columns while ignoring null values and applying lemmatization
    def concatenate_and_lemmatize(row):
        columns = ['product_name', 'category']
        values = [str(row[col]) for col in columns if pd.notnull(row[col])]
        concatenated_string = ' '.join(values)

        # Tokenize the concatenated string
        tokens = word_tokenize(concatenated_string)

        # Apply lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

        # Join lemmatized tokens back into a single string
        lemmatized_string = ' '.join(lemmatized_tokens)

        return lemmatized_string

    # Apply the function to concatenate columns and lemmatize text
    df_Ama['lemmatized_concatenated'] = df_Ama.apply(concatenate_and_lemmatize, axis=1)

    # Save the result back to a CSV file
    df_Ama.to_csv('output_new.csv', index=False)

    df_lemmed = pd.read_csv('output_new.csv')

    column_name = 'category'

    # Function to extract unique words from a column
    def extract_unique_words(column):
        unique_words = set()
        for text in column.dropna():  # Drop NaN values to avoid errors
            tokens = word_tokenize(str(text))  # Tokenize the text
            unique_words.update(tokens)  # Add tokens to the set
        return unique_words

    # Extract unique words from the specified column
    unique_words = extract_unique_words(df_lemmed[column_name])

    # Specify the column to search for unique values
    column_name = 'lemmatized_concatenated' 

    # Function to count occurrences of each unique word in a column
    def count_word_occurrences(column, unique_words):
        word_counts = {word: 0 for word in unique_words}
        for text in column.dropna():  # Drop NaN values to avoid errors
            tokens = word_tokenize(str(text))  # Tokenize the text
            for token in tokens:
                if token in word_counts:
                    word_counts[token] += 1
        return word_counts

    # Get the word occurrences
    word_occurrences = count_word_occurrences(df_lemmed[column_name], unique_words)

    # Define a function to retrieve the lemmatized value for a given title
    def get_lemmatized_value(title):
        title = title.strip().lower()  # Remove leading and trailing whitespaces and convert to lowercase
        return df_lemmed.loc[df_lemmed['product_name'].str.strip().str.lower() == title, 'lemmatized_concatenated'].values[0]
        return matching_rows.values[0] if not matching_rows.empty else None

    # Example usage
    selected_title = "Shopping Cart"
    lemmatized_value = get_lemmatized_value(selected_title)
    print("Lemmatized value for the selected title:")
    print(lemmatized_value)

    # Define a function to extract unique words from a string
    def extract_unique_words(text):
        tokens = word_tokenize(str(text))  # Tokenize the text
        unique_words = set(tokens)  # Convert tokens to a set to get unique words
        return unique_words

    # Extract unique words from the lemmatized value
    unique_words = extract_unique_words(lemmatized_value)
    print("Unique words found in the lemmatized value:")
    print(unique_words)

    # Define a function to check if a unique word is present in the lemmatized value
    def is_unique_word_present(unique_word, lemmatized_value):
        tokens = word_tokenize(str(lemmatized_value))  # Tokenize the lemmatized value
        return unique_word.lower() in [token.lower() for token in tokens]
        return unique_word.lower() in [token.lower() for token in tokens]

    # Example usage
    unique_word_to_check = "Groceries"  # Change this to the unique word you want to check
    is_present = is_unique_word_present(unique_word_to_check, lemmatized_value)


    # Function to count occurrences of a unique word in a column
    def count_word_occurrences(column, unique_word):
        word_count = 0
        matching_rows = []
        for text in column.dropna():  # Drop NaN values to avoid errors
            tokens = word_tokenize(str(text))  # Tokenize the text
            if unique_word.lower() in [token.lower() for token in tokens]:
                word_count += 1
                matching_rows.append(text)
        return word_count, matching_rows

    if is_present:
        column_name = 'lemmatized_concatenated'
        word_count, matching_rows = count_word_occurrences(df_lemmed[column_name], unique_word_to_check)
        print(f"\n'{unique_word_to_check}' appears {word_count} times in the '{column_name}' column.")
        
        print("\nRandom 10 lemmatized values containing the unique word:")
        random_sample = random.sample(matching_rows, min(9, len(matching_rows)))
        recommended_titles = df_lemmed[df_lemmed[column_name].isin(random_sample)]['product_name'].values
        
        print("\nThe first 10 recommendations are:")
        for i, title in enumerate(recommended_titles, start=1):
            print(f"{i}: {title}")

    # Function to count occurrences of a unique word in a column and return rows containing that word
    def count_word_occurrences(column, unique_word):
        word_count = 0
        matching_rows = []
        for index, text in column.dropna().items():  # Drop NaN values to avoid errors
            tokens = [token.lower() for token in word_tokenize(str(text))]  # Tokenize the text and make lowercase
            if unique_word.lower() in tokens:
                word_count += 1
                matching_rows.append(index)
        return word_count, matching_rows  

    # Example input from the user
    #titles = input("Enter titles separated by commas: ").split(',')
    titles=entry.split(',')

    all_recommendations = set()

    # Iterate through each title provided by the user
    for selected_title in titles:
        selected_title = selected_title.strip()  # Remove leading and trailing whitespaces
        lemmatized_value = get_lemmatized_value(selected_title)
        
        # Extract unique words from the lemmatized value
        unique_words = extract_unique_words(lemmatized_value)
        
        # Iterate through each unique word to find matching rows
        for unique_word in unique_words:
            word_count, matching_rows = count_word_occurrences(df_lemmed['lemmatized_concatenated'], unique_word)
            
            # Collect recommendations
            all_recommendations.update(df_lemmed.loc[matching_rows, 'product_name'].tolist())


    # Remove the selected titles from the recommendations
    all_recommendations.difference_update(titles)

    # Select 10 random recommendations
    recommendations_list = random.sample(sorted(all_recommendations), min(len(all_recommendations), 10))

    print("\nFirst 10 recommendations:")
    for recommendation in recommendations_list:
        print(recommendation)

    return recommendations_list

#get_recommendations('Nestle Choccolim')

def get_recommendations(entry):
    df = pd.read_csv(url)
    print(url)
    product=df.loc[entry-1]['product_name']
    print(product)
    try:
        recommendations_list=get_list(product)
        message=f'Extracted {len(recommendations_list)} recommendations from {product}'
    except:
        message='Error'
        recommendations_list=[]


    return recommendations_list,message

get_recommendations(5)