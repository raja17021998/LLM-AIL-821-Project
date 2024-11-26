import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


train_data = pd.read_csv('/home/pooja/shashwat/LLM-Project/guj_train.csv')
val_data = pd.read_csv('/home/pooja/shashwat/LLM-Project/guj_valid.csv')

def translate_text_with_progress(df, text_column, label_column):
    """
    Translates a specified text column from Gujarati to English using deep-translator.
    Returns a DataFrame containing valid translations only.
    """
    original_texts, translated_texts, labels = [], [], []
    for idx, text in tqdm(enumerate(df[text_column]), total=len(df), desc="Translating"):
        try:
            
            translation = GoogleTranslator(source='gu', target='en').translate(text)
            original_texts.append(text) 
            translated_texts.append(translation)  
            labels.append(df[label_column].iloc[idx])  
        except Exception as e:
            print(f"Error translating text: {text}. Error: {e}") 
            continue  
    
    return pd.DataFrame({'label': labels, 'original_text': original_texts, 'translated_text': translated_texts})


print("Translating training data...")
train_translated = translate_text_with_progress(train_data, 'headline', 'label')


print("Translating validation data...")
val_translated = translate_text_with_progress(val_data, 'headline', 'label')


train_output_path = '/home/pooja/shashwat/LLM-Project/eng_guj_train.csv'
val_output_path = '/home/pooja/shashwat/LLM-Project/eng_guj_val.csv'

train_translated.to_csv(train_output_path, index=False)
val_translated.to_csv(val_output_path, index=False)

print(f"Train data saved to {train_output_path}")
print(f"Validation data saved to {val_output_path}")
