from deep_translator import GoogleTranslator
import pandas as pd

file_path = "English_French_Hindi_words.csv" 
df = pd.read_csv(file_path)


translator_french = GoogleTranslator(source='en', target='fr')
translator_hindi = GoogleTranslator(source='en', target='hi')

def translate_word(word, translator):
    try:
        return translator.translate(word)
    except Exception as e:
        return "Error"

df['French'] = df['English'].apply(lambda x: translate_word(x, translator_french))
df['Hindi'] = df['English'].apply(lambda x: translate_word(x, translator_hindi))

output_path = "/home/pooja/shashwat/LLM-Project/fully_translated_english_french_hindi.csv"
df.to_csv(output_path, index=False)

print(f"Translated CSV saved at: {output_path}")
