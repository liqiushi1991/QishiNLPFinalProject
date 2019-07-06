import torch
import os

import torch.nn as nn
import torchvision.models as models
import numpy as np

from Measures      import AverageMeter, accuracy, calculate_caption_lengths
from DataLoader    import get_loader
from model         import Encoder, Decoder
from Config        import config
from BuildVocab    import build_and_save_vocab


# Device configuration
device = torch.device('cpu')


def train():
    model_path = config.get('model_path', './model/')
    log_step = config.get('log_step', 10)
    save_step = config.get('save_step', 1000)
    hidden_size = config.get('decoder_hidden_size', 512)
    num_epochs = config.get('num_epochs', 5)
    learning_rate = config.get('learning_rate', 0.001)
    alpha_c = config.get('alpha_c', 1)

    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Used for calculating bleu scores
    references = []
    hypotheses = []

    # Load vocabulary
    vocab = build_and_save_vocab()

    # Build data loader
    data_loader = get_loader('train')

    # Build the models
    encoder = Encoder(config.get('image_net')).to(device)
    decoder = Decoder(encoder.dim, len(vocab), hidden_size=hidden_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            # Forward, backward and optimize
            features = encoder.forward(images)
            prediction, alphas = decoder.forward(features, captions)

            att_regularization = alpha_c * ((1 - alphas.sum(1)) ** 2).mean()
            loss = criterion(prediction.permute(0, 2, 1), captions) + att_regularization
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            total_caption_length = calculate_caption_lengths(vocab.word2idx, captions)
            acc1 = accuracy(prediction.permute(0, 2, 1), captions, 1)
            acc5 = accuracy(prediction.permute(0, 2, 1), captions, 5)
            losses.update(loss.item(), total_caption_length)
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)

            # Print log info
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                print(
                    'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f}), Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                        top1=top1, top5=top5))
            # Save the model checkpoints
            if (i + 1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
