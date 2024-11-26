


import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


file_path = "fully_translated_english_french_hindi.csv"
dataset = pd.read_csv(file_path)


model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embeddings(word, tokenizer, model):
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


languages = ["English", "French", "Hindi"]
language_colors = {"English": "red", "French": "blue", "Hindi": "orange"}


embeddings = []
labels = []
original_words = []
for index, row in dataset.iterrows():
    for lang in languages:
        word = row[lang]
        embedding = get_embeddings(word, tokenizer, model)
        embeddings.append(embedding)
        labels.append(lang)  
        original_words.append(row["English"])  


pca_3d = PCA(n_components=3)
reduced_embeddings_3d = pca_3d.fit_transform(np.array(embeddings))


scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_embeddings_3d = scaler.fit_transform(reduced_embeddings_3d)


num_clusters = 5 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_embeddings_3d)


cosine_similarities = []
for i in range(0, len(embeddings), 3):  
    english_embedding = embeddings[i]
    french_embedding = embeddings[i + 1]
    hindi_embedding = embeddings[i + 2]

    
    similarity_english_french = cosine_similarity([english_embedding], [french_embedding])[0][0]
    similarity_english_hindi = cosine_similarity([english_embedding], [hindi_embedding])[0][0]
    similarity_french_hindi = cosine_similarity([french_embedding], [hindi_embedding])[0][0]

    cosine_similarities.append({
        "word": original_words[i // 3],
        "English-French": similarity_english_french,
        "English-Hindi": similarity_english_hindi,
        "French-Hindi": similarity_french_hindi
    })


similarity_df = pd.DataFrame(cosine_similarities)
similarity_output_path = "cosine_similarity_results.csv"
similarity_df.to_csv(similarity_output_path, index=False)
print(f"Cosine similarity results saved to {similarity_output_path}")


cluster_colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


for i, embedding in enumerate(normalized_embeddings_3d):
    cluster_id = clusters[i]
    ax.scatter(embedding[0], embedding[1], embedding[2], color=cluster_colors[cluster_id], alpha=0.7)
    ax.text(embedding[0] + 0.02, embedding[1] + 0.02, embedding[2] + 0.02, original_words[i // 3], fontsize=9)


for cluster_center in kmeans.cluster_centers_:
    ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], color="black", s=100, marker="x")


language_legend = [Line2D([0], [0], marker='o', color=color, markersize=10, label=f"{lang}") 
                   for lang, color in language_colors.items()]

ax.legend(handles=language_legend, loc="upper left", title="Languages")
ax.set_title("3D Word Embeddings in Vector Space with Clusters")
ax.set_xlabel("Principal Component 1 (Normalized)")
ax.set_ylabel("Principal Component 2 (Normalized)")
ax.set_zlabel("Principal Component 3 (Normalized)")


output_dir = "/home/pooja/shashwat/LLM-Project/plots/task1"
os.makedirs(output_dir, exist_ok=True)
clustered_3d_plot_path = os.path.join(output_dir, "3d_word_embeddings_with_clusters.png")
plt.savefig(clustered_3d_plot_path, format="png", dpi=300)
plt.show()

print(f"3D plot saved to {clustered_3d_plot_path}")

