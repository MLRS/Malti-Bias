# Install necessary packages
# !pip install transformers
# !pip install torch
# !pip install sklearn
# !pip install matplotlib

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the Maltese BERT model and tokenizer from Hugging Face (BERTu) using the base 'bert-base-multilingual-cased' model as a proxy for BERTu
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("MLRS/BERTu")
model = AutoModelForMaskedLM.from_pretrained("MLRS/BERTu",output_hidden_states=True)

gendered_words_mt = [ "għalliem", "għalliema",
    "qawwi",         # strong
    "dgħajjef",       # weak
    "kuraġġuż",      # courageous
    "tħajjir",        # inspiring
    "gentili",        # kind
    "brutali",        # brutal
    "intelliġenti",   # intelligent
    "ikrah",          # hateful
    "sensittiv",      # sensitive
    "tajjeb",         # good
    "kattiv",         # bad
    "kompetenti",     # competent
    "inkompetenti",   # incompetent
    "eċċellenti",     # excellent
    "normali",        # normal
    "professjonali",  # professional
    "domestiku",      # domestic
    "karitatevoli",   # charitable
    "indifferenti",   # indifferent
    "solitarju",      # solitary
    "soċjali"         # social
]


# Function to get word embeddings from BERT
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors='pt', add_special_tokens=False)
    with torch.no_grad():
        outputs = model(**inputs)
        # Check if the model output has hidden states
        if hasattr(outputs, 'hidden_states'):
            # Extract the last hidden state
            return outputs.hidden_states[-1].mean(dim=1).squeeze(0).numpy()
        else:
            # If the model doesn't return hidden states, check for 'logits'
            return outputs.logits.mean(dim=1).squeeze(0).numpy()


# Get embeddings
hu_embedding = get_word_embedding("għalliem")
hi_embedding = get_word_embedding("għalliema")

# Create a direction vector from 'hu' to 'hi'
direction_vector = hi_embedding - hu_embedding

# Normalize the direction vector
direction_vector = direction_vector / np.linalg.norm(direction_vector)

#Project other gendered word embeddings onto the 'hu-hi' axis
def project_onto_direction(word_embedding, direction_vector):
    # Calculate the dot product to get the scalar projection onto the direction vector
    return np.dot(word_embedding - hu_embedding, direction_vector)

# Project onto an arbitrary vertical axis for separation (optional, here I use random for vertical separation)
def random_vertical_projection():
    return np.random.uniform(-1, 1)  # Random vertical position for better visualization

# Get the projections for all gendered words (Maltese)
horizontal_projections = []
vertical_projections = []
for word in gendered_words_mt:
    word_embedding = get_word_embedding(word)
    # Projection along the 'hu-hi' axis (horizontal)
    horizontal_projection = project_onto_direction(word_embedding, direction_vector)
    horizontal_projections.append(horizontal_projection)
    # Random vertical projection (you can modify this to use another dimension of the embedding)
    vertical_projection = random_vertical_projection()
    vertical_projections.append(vertical_projection)

# Convert projections to NumPy arrays
horizontal_projections = np.array(horizontal_projections)
vertical_projections = np.array(vertical_projections)

# Plot the words along the male-female axis with vertical separation
plt.figure(figsize=(10, 6))

# Plot all words except male and female vars
plt.scatter(horizontal_projections[2:], vertical_projections[2:], color='blue')

# Highlight male and female variables with different colors and larger marker sizes
hu_index = gendered_words_mt.index("għalliem")
hi_index = gendered_words_mt.index("għalliema")

plt.scatter(horizontal_projections[hu_index], vertical_projections[hu_index], color='red', s=100, label='għalliem')   # Red for 'hu'
plt.scatter(horizontal_projections[hi_index], vertical_projections[hi_index], color='green', s=100, label='għalliema')  # Green for 'hi'

# Annotate each point with the corresponding word
for i, word in enumerate(gendered_words_mt):
    plt.annotate(word, (horizontal_projections[i], vertical_projections[i]), fontsize=12, ha='center', va='bottom')

# Remove the vertical dotted line (axvline)
# plt.axvline(0, color='black', linestyle='--')  # Commented out to remove the line

# Plot title and show the plot
plt.title("Projection of Gendered Words in Maltese on the 'għalliem-għalliema' Axis with Vertical Separation")
plt.legend()
plt.show()
