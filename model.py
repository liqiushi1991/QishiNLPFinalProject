import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import init


class Encoder(nn.Module):
    def __init__(self, encoder=models.resnet50):
        if encoder is None:
            raise Exception('The model is None.')
        super(Encoder, self).__init__()
        if 'resnet' in encoder.__module__:
            resnet = encoder(pretrained=True)
            modules = list(resnet.children())[:-2]
            self.dim = 2048
        elif 'vgg' in encoder.__module__:
            resnet = encoder(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.dim = 512
        else:
            raise NotImplementedError
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)  # [batch, self.dim, 7, 7]
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))  # [batch, 49, self.dim=2048]
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_size=512):
        super(Attention, self).__init__()

        self.affine_W = nn.Linear(encoder_dim, hidden_size)
        self.affine_U = nn.Linear(hidden_size, hidden_size)

        self.affine_V = nn.Linear(hidden_size, 1)

    def init_weights(self):
        init.xavier_uniform(self.affine_W.weight)
        init.xavier_uniform(self.affine_U.weight)
        init.xavier_uniform(self.affine_V.weight)

    def forward(self, a, prev_hidden_state):
        att = torch.tanh(self.affine_W(a) + self.affine_U(prev_hidden_state).unsqueeze(1))  # [batch, 49, 1]
        e_t = self.affine_V(att).squeeze(2)

        alpha_t = nn.Softmax(1)(e_t)  # [batch, 49]
        context_t = (a * alpha_t.unsqueeze(2)).sum(1)  # [batch, 2048]

        return context_t, alpha_t


class Decoder(nn.Module):
    def __init__(self, encoder_dim, vocab_size, hidden_size=512):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim

        self.init_affine_h = nn.Linear(encoder_dim, hidden_size)
        self.init_affine_c = nn.Linear(encoder_dim, hidden_size)

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = Attention(encoder_dim, hidden_size=hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.lstm = nn.LSTMCell(hidden_size + encoder_dim, hidden_size)
        self.output_W = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        batch_size = features.size(0)

        features_avg = features.mean(dim=1)
        h = torch.tanh(self.init_affine_h(features_avg))
        c = torch.tanh(self.init_affine_c(features_avg))

        T = max([len(caption) for caption in captions])
        prev_word = torch.zeros(batch_size, 1).long()
        pred_words = torch.zeros(batch_size, T, self.vocab_size)  # [128, 26, 2699]
        alphas = torch.zeros(batch_size, T, features.size(1))

        embedding = self.embedding(prev_word)

        for t in range(T):
            context_t, alpha_t = self.attention.forward(features, h)
            gate = torch.sigmoid(self.f_beta(h))
            gated_context = gate * context_t

            # dim = 3 is for mini batch
            embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
            lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = torch.sigmoid(self.output_W(h))

            pred_words[:, t] = output
            alphas[:, t] = alpha_t
            if not self.training:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
            else:
                prev_word = captions[:, t]
                embedding = self.embedding(prev_word)
        return pred_words, alphas

    def caption(self, features, beam_size):
        '''
        From https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        '''

        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        features_avg = features.mean(dim=1)
        h = torch.tanh(self.init_affine_h(features_avg))
        c = torch.tanh(self.init_affine_c(features_avg))

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = torch.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            features = features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha
