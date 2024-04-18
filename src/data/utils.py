def load_data(path: str, lowercase: bool = True):
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(line.rstrip("\n"))
    return [sentence.lower() for sentence in corpus] if lowercase else corpus
