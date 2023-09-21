from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
import numpy as np

while True:
    sentences = [
        'I would pick up the blue key',
        'Pick up red key'
    ]

    sentences = [input("Sentence A: "), input("Sentence B: ")]

    #Sentences are encoded by calling model.encode()
    a, b = model.encode(sentences)

    cos_sim = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    print(cos_sim)
