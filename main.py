from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import tensorflow as tf

keywords = open("keywords.txt", "r").read().split("\n")

print(keywords)


# Convert keywords into a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(keywords)

# TensorFlow KMeans clustering
num_clusters = 20  # Adjust this as needed
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)

# Train the KMeans model
def input_fn():
    return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(X.toarray(), dtype=tf.float32), num_epochs=1)

kmeans.train(input_fn)

# Get the cluster assignments
cluster_indices = list(kmeans.predict_cluster_index(input_fn))

# Group keywords based on cluster assignments
clusters = {}
for i, cluster_idx in enumerate(cluster_indices):
    if cluster_idx not in clusters:
        clusters[cluster_idx] = [keywords[i]]
    else:
        clusters[cluster_idx].append(keywords[i])

result = ""

with open("out.txt", "w") as out:
    count = 0
    for i in clusters:
        result += f"cluster {count + 1}:\n-----------------------\n"
        for keyword in clusters[i]:
            result += keyword + "\n"
        result += "\n"
        count += 1

    out.write(result)