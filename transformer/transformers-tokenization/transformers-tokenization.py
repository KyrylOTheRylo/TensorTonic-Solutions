class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        self.id_to_word = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>",
        }

        self.vocab_size = 4

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        words = []

        for text in texts:
            words.extend(text.lower().split())

        words = sorted(set(words))

        for word in words:
            if word not in self.word_to_id:
                idx = self.vocab_size

                self.word_to_id[word] = idx
                self.id_to_word[idx] = word

                self.vocab_size += 1

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        words = text.lower().split()

        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in words
        ]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        words = []

        for idx in ids:
            words.append(
                self.id_to_word.get(idx, self.unk_token)
            )

        return " ".join(words)          