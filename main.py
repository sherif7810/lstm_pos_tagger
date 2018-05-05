import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

WORD_EMBEDDING_DIM = 5
CHAR_EMBEDDING_DIM = 6
CHAR_REPR_DIM = 3
HIDDEN_DIM = 6

word_to_ix = {}
char_to_ix = {}
tag_to_ix = {}
for seq, tags in training_data:
    for word in seq:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)

    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)


def make_ixs(seq, to_ix):
    ixs = torch.tensor([to_ix[w] for w in seq])
    return ixs


class LSTMTagger(nn.Module):
    """LSTM part-os-speech (PoS) tagger."""

    def __init__(self, word_embedding_dim, char_embedding_dim, char_repr_dim, hidden_dim, vocab_size, chars_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.char_repr_dim = char_repr_dim
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(chars_size, char_embedding_dim)

        self.char_lstm = nn.LSTM(char_embedding_dim, char_repr_dim)
        self.word_lstm = nn.LSTM(word_embedding_dim + char_repr_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.char_lstm_hidden = (torch.zeros(1, 1, self.char_repr_dim),
                                 torch.zeros(1, 1, self.char_repr_dim))
        self.word_lstm_hidden = (torch.zeros(1, 1, self.hidden_dim),
                                 torch.zeros(1, 1, self.hidden_dim))

    def init_word_hidden(self):
        """Initialise word LSTM hidden state."""
        self.word_lstm_hidden = (torch.zeros(1, 1, self.hidden_dim),
                                 torch.zeros(1, 1, self.hidden_dim))

    def init_char_hidden(self):
        """Initialise character LSTM hidden state."""
        self.char_lstm_hidden = (torch.zeros(1, 1, self.char_repr_dim),
                                 torch.zeros(1, 1, self.char_repr_dim))

    def forward(self, sentence):
        sentence_length = len(sentence)
        word_characters_ixs = {}
        for word in sentence:
            word_ix = torch.tensor([word_to_ix[word]])
            char_ixs = make_ixs(word, char_to_ix)
            word_characters_ixs[word_ix] = char_ixs

        word_embeds = []
        for word_ix, char_ixs in word_characters_ixs.items():
            word_embed = self.word_embeddings(word_ix)

            self.init_char_hidden()
            c = None  # Character-level representation.
            for char_ix in char_ixs:
                char_embed = self.char_embeddings(char_ix)
                c, self.char_lstm_hidden = self.char_lstm(
                    char_embed.view(1, 1, -1), self.char_lstm_hidden)
            word_embeds.append(word_embed)
            word_embeds.append(c.view(1, -1))
        word_embeds = torch.cat(word_embeds, 1)

        lstm_out, self.word_lstm_hidden = self.word_lstm(
            word_embeds.view(sentence_length, 1, -1), self.word_lstm_hidden)
        tag_space = self.hidden2tag(lstm_out.view(sentence_length, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(WORD_EMBEDDING_DIM,
                   CHAR_EMBEDDING_DIM, CHAR_REPR_DIM,
                   HIDDEN_DIM,
                   len(word_to_ix), len(char_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
for epoch in range(300):
    total_loss = torch.tensor(0.)

    for sentence, tags in training_data:
        targets = make_ixs(tags, tag_to_ix)

        model.zero_grad()

        model.init_word_hidden()
        tag_scores = model(sentence)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss

    print('Epoch {}: Loss = {}.'.format(epoch + 1, loss.item()))

# Testing
print('Testing:')
with torch.no_grad():
    inputs = training_data[0][0]

    model.init_word_hidden()
    tag_scores = model(inputs)

    print(tag_scores)
