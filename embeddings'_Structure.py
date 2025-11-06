import torch

embedding_matrix = torch.load("word2vec_embeddings.pt", map_location="cpu")
vocab = torch.load("vocab_list.pt", map_location="cpu")
embedding = {}
for token, idx in vocab.items():    # works only if vocab is dict token→index
    embedding[token] = embedding_matrix[idx].numpy()
    
print(embedding["peter"])
