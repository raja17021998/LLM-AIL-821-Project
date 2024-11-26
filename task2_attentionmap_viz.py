import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, BloomForSequenceClassification, AutoModelForSequenceClassification

from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import cosine_similarity

warnings.filterwarnings('ignore')

# # Load tokenizer and model checkpoint
# model_checkpoint = "./results/lora/checkpoint-1890"
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")
# model = BloomForSequenceClassification.from_pretrained(model_checkpoint,output_hidden_states=True)

model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BloomForSequenceClassification.from_pretrained(model_name,output_hidden_states=True)
model.eval()

def tokenize_input(sentence):
    return tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

# 1. Sentiment Analysis
def infer_sentiment(sentence):
    inputs = tokenize_input(sentence)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment


sentence= "Sad news! After all you got admitted in IIT Delhi"
sentiment = infer_sentiment(sentence)
print(f"Sentiment: {sentiment}")



def extract_attention(sentence):

    inputs = tokenize_input(sentence)

    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]

    tokens = [
        tokenizer.convert_ids_to_tokens(token_id.item()).replace("Ġ", "").replace("▁", "")
        for token_id, mask in zip(input_ids, attention_mask)
        if mask == 1
    ]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)


    attentions = [
        layer[:, :, :len(tokens), :len(tokens)] for layer in outputs.attentions
    ]

    return attentions, tokens

def plot_attention(attention, tokens, layer=0, head=0, title="Attention Map", save_path=None):

    attn_map = attention[layer][0, head].detach().cpu().numpy()

    plt.figure(figsize=(14, 12)) 
    sns.heatmap(
        attn_map,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        square=True,
        annot=False,
        fmt=".2f",
    )
    plt.title(f"{title} - Layer {layer}, Head {head}", fontsize=16)
    plt.xlabel("Key Tokens", fontsize=12, labelpad=20)  
    plt.ylabel("Query Tokens", fontsize=12, labelpad=20)  
    plt.xticks(rotation=90, ha="right", fontsize=7.5)  
    plt.yticks(fontsize=7.5)

 
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight", pad_inches=1)
        print(f"Attention map saved to: {save_path}")

    plt.show()

def aggregate_attention(attention, tokens, layer, save_path=None):

    aggregated_map = attention[layer].mean(dim=1).squeeze(0).detach().cpu().numpy()

    plt.figure(figsize=(14, 12)) 
    sns.heatmap(
        aggregated_map,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Purples",
        square=True,
        annot=False,
        fmt=".2f",
    )
    plt.title(f"Aggregated Attention - Layer {layer}", fontsize=16)
    plt.xlabel("Key Tokens", fontsize=12, labelpad=20)  
    plt.ylabel("Query Tokens", fontsize=12, labelpad=20)  
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.3)

    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight", pad_inches=1)
        print(f"Aggregated attention map saved to: {save_path}")

    plt.show()


# Attention visualization for a user-provided sentence
print("\n-- Attention Visualization --")
sentence_for_attention = (
    "Artificial intelligence is transforming every aspect of our lives, from healthcare to education, "
    "as it enables doctors to predict diseases with greater accuracy, automates repetitive tasks in industries, "
    "enhances personalized learning experiences for students, revolutionizes transportation through autonomous vehicles, "
    "optimizes energy consumption in smart cities, and even powers entertainment platforms by providing tailored recommendations, "
    "all while raising critical questions about ethics, privacy, and security, demanding thoughtful regulation and oversight "
    "to ensure that AI technologies are deployed responsibly, equitably, and inclusively, fostering innovation that benefits humanity, "
    "while avoiding potential pitfalls such as algorithmic bias, misuse of sensitive data, and over-reliance on machines, "
    "which could diminish critical human decision-making skills, leading to societal dependencies that may be difficult to reverse, "
    "requiring collaboration among governments, industries, and academia to strike a balance between progress and caution, "
    "ensuring AI remains a tool for good and not a source of harm."
)


attention, tokens = extract_attention(sentence_for_attention)



layers=[2,3,7,12,16]
heads= [1,2,3,4]

for layer in layers:
    for head in heads:
        
        os.makedirs("plots/task2/attention_maps/english", exist_ok=True)
        plot_attention(
            attention,
            tokens,
            layer=layer,
            head=head,
            title="Attention Map - Sentence",
            save_path=f"plots/task2/attention_maps/english/attention_layer{layer}_head{head}.jpg",
        )
        aggregate_attention(
            attention,
            tokens,
            layer=3,
            save_path=f"plots/task2/attention_maps/english/aggregated_attention_layer{layer}.jpg",
        )
        
        
        
sentence_for_attention = (
    "ਪੰਛੀ ਅਸਮਾਨ ਵਿੱਚ ਉੱਡਦਾ ਹੈ, ਉਹ ਸੁਤੰਤਰਤਾ ਅਤੇ ਆਜ਼ਾਦੀ ਦਾ ਪ੍ਰਤੀਕ ਹੈ। ਉਸਦੀ ਉੱਡਾਣ ਉਸਦੇ ਦਿਲ ਦੀ ਖੁਸ਼ੀ ਅਤੇ ਬੇਫਿਕਰੀ ਨੂੰ ਦਰਸਾਉਂਦੀ ਹੈ, ਪਰ ਜਦੋਂ ਉਹ ਧਰਤੀ ਉੱਤੇ ਆਉਂਦਾ ਹੈ, ਤਾਂ ਉਹ ਅੰਨ ਅਤੇ ਪਾਣੀ ਦੀ ਤਲਾਸ਼ ਕਰਦਾ ਹੈ। ਇਹ ਸਾਡੇ ਲਈ ਇਹ ਸਿਖਲਾਈ ਹੈ ਕਿ ਜੀਵਨ ਵਿੱਚ ਆਜ਼ਾਦੀ ਮਹੱਤਵਪੂਰਣ ਹੈ, ਪਰ ਇਸਦੇ ਨਾਲ ਜ਼ਰੂਰਤਾਂ ਨੂੰ ਪੂਰਾ ਕਰਨਾ ਵੀ ਲਾਜ਼ਮੀ ਹੈ। ਅਸਮਾਨ ਦੀ ਖੁਦਮੁਖਤੀ ਉਸਦੇ ਆਤਮਵਿਸ਼ਵਾਸ ਨੂੰ ਵਧਾਉਂਦੀ ਹੈ, ਜਦਕਿ ਧਰਤੀ ਦੀ ਮਿੱਟੀ ਉਸਦੇ ਜੀਵਨ ਦੀ ਮੂਲ ਭੂਮਿਕਾ ਨੂੰ ਸਮਰਪਿਤ ਕਰਦੀ ਹੈ। ਸਾਡੇ ਜੀਵਨ ਵਿੱਚ ਵੀ ਇੱਥੇ ਸੁਤੰਤਰਤਾ ਅਤੇ ਆਵਸ਼ਕਤਾਵਾਂ ਦੇ ਵਿਚਕਾਰ ਸੰਤੁਲਨ ਬਣਾਈ ਰੱਖਣ ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ, ਤਾਂ ਜੋ ਅਸੀਂ ਪ੍ਰਗਤੀ ਅਤੇ ਤ੍ਰਿਪਤੀ ਦੋਵੇਂ ਹਾਸਲ ਕਰ ਸਕੀਏ।"
)


attention, tokens = extract_attention(sentence_for_attention)



layers=[3,7,12,15]
heads= [1,2,3,4]

for layer in layers:
    for head in heads:
        
        os.makedirs("plots/task2/attention_maps/punjabi", exist_ok=True)
        plot_attention(
            attention,
            tokens,
            layer=layer,
            head=head,
            title="Attention Map - Sentence",
            save_path=f"plots/task2/attention_maps/punjabi/attention_layer{layer}_head{head}.jpg",
        )
        aggregate_attention(
            attention,
            tokens,
            layer=3,
            save_path=f"plots/task2/attention_maps/punjabi/aggregated_attention_layer{layer}.jpg",
        )

    


