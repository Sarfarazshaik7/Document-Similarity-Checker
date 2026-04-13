from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Enter file paths
file1 = "doc1.txt"
file2 = "doc2.txt"

# Step 2: Read files
with open(file1, 'r', encoding='utf-8') as f:
    doc1 = f.read()

with open(file2, 'r', encoding='utf-8') as f:
    doc2 = f.read()

# Step 3: TF-IDF
documents = [doc1, doc2]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 4: Cosine similarity
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Step 5: Output
score = similarity[0][0]
print("\nSimilarity Score:", round(score * 100, 2), "%")

if score > 0.7:
    print("Result: Documents are Highly Similar")
elif score > 0.4:
    print("Result: Documents are Moderately Similar")
else:
    print("Result: Documents are Not Similar")
