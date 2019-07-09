import torch

import numpy              as np
import torch.nn           as nn
import torchvision.models as models


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


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
        self.dim = self.dim // 2
        self.affine_W = nn.Linear(self.dim * 2, self.dim)
        self.reset_parameters()

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)  # [batch, self.dim, 8, 8]
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))  # [batch, 64, self.dim=2048]
        features = self.affine_W(features)
        return features

    def reset_parameters(self):
        self.affine_W.weight.data.uniform_(*hidden_init(self.affine_W))


class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_size=512):
        super(Attention, self).__init__()

        self.affine_W = nn.Linear(encoder_dim, hidden_size)
        self.affine_U = nn.Linear(hidden_size, hidden_size)

        self.affine_V = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.affine_W.weight.data.uniform_(*hidden_init(self.affine_W))
        self.affine_U.weight.data.uniform_(*hidden_init(self.affine_U))
        self.affine_V.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, a, prev_hidden_state):
        att = torch.tanh(self.affine_W(a) + self.affine_U(prev_hidden_state).unsqueeze(1))  # [batch, 49, 1]
        e_t = self.affine_V(att).squeeze(2)

        alpha_t = nn.Softmax(1)(e_t)  # [batch, 64]
        context_t = (a * alpha_t.unsqueeze(2)).sum(1)  # [batch, 1024]

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

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.weight.data.uniform_(-3e-3, 3e-3)
        self.init_affine_h.weight.data.uniform_(*hidden_init(self.init_affine_h))
        self.init_affine_c.weight.data.uniform_(*hidden_init(self.init_affine_c))
        self.f_beta.weight.data.uniform_(*hidden_init(self.f_beta))
        self.output_W.weight.data.uniform_(*hidden_init(self.output_W))

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
