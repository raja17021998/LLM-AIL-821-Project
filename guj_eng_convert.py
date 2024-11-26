import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


train_data = pd.read_csv('/home/pooja/shashwat/LLM-Project/guj_train.csv')
val_data = pd.read_csv('/home/pooja/shashwat/LLM-Project/guj_valid.csv')

def translate_text_with_progress(df, column_name):

    ben_text, eng_text, sentiments = [], [], []
    for idx, text in tqdm(enumerate(df[column_name]), total=len(df), desc="Translating"):
        try:
            
            translation = GoogleTranslator(source='gu', target='en').translate(text)
            ben_text.append(text)  
            eng_text.append(translation)  
            sentiments.append(df['sentiment'].iloc[idx])  
        except Exception as e:
            print(f"Error translating text: {text}. Error: {e}")  
            continue  
    
    return pd.DataFrame({'sentiment': sentiments, 'ben_text': ben_text, 'eng_text': eng_text})

print("Translating training data...")
train_translated = translate_text_with_progress(train_data, 'text')

print("Translating validation data...")
val_translated = translate_text_with_progress(val_data, 'text')


train_output_path = 'eng_guj_train.csv'
val_output_path = 'eng_guj_val.csv'

train_translated.to_csv(train_output_path, index=False)
val_translated.to_csv(val_output_path, index=False)

print(f"Train data saved to {train_output_path}")
print(f"Validation data saved to {val_output_path}")
