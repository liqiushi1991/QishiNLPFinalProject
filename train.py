import torch
import os
import torch.optim

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


def adjust_learning_rate(optimizer, decay_rate):
    """
    decay the learning rate by a decay factor
    :param optimizer: optimizer with learning rate to be decayed
    :param decay_rate: decaying factor the learning rate
    :return:
    """
    print(f'\nDecaying learning rate by {decay_rate}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    print(f'Updated learning rate to {optimizer.param_groups[0]["lr"]}\n')


def train():
    model_path = config.get('model_path', './model/')
    log_step = config.get('log_step', 10)
    hidden_size = config.get('decoder_hidden_size', 512)
    num_epochs = config.get('num_epochs', 5)
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
    data_loader_valid = get_loader('validate')

    # Build the models
    if config.get('checkpoint') is None:
        epochs_since_improvement = config.get('epochs_since_improvement')
        best_score = 0.
        encoder = Encoder(config.get('image_net')).to(device)
        decoder = Decoder(encoder.dim, len(vocab), hidden_size=hidden_size).to(device)

        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=config.get('encoder_lr', 1e-4))

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=config.get('decoder_lr', 1e-3))
    else:
        checkpoint = torch.load(config.get('checkpoint'))
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_score = checkpoint['best_score']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the models
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        if epochs_since_improvement == 20:
            print('Reached the max epochs_since_improvement. Training is done.')
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.9)
            adjust_learning_rate(encoder_optimizer, 0.9)

        for i, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)

            # Forward, backward and optimize
            features = encoder.forward(images)
            prediction, alphas = decoder.forward(features, captions)

            att_regularization = alpha_c * ((1 - alphas.sum(1)) ** 2).mean()
            loss = criterion(prediction.permute(0, 2, 1), captions) + att_regularization

            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()

            loss.backward()

            decoder_optimizer.step()
            encoder_optimizer.step()

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

            valid_score = validate(data_loader_valid, encoder, decoder, criterion, vocab)
            if valid_score >= best_score:
                epochs_since_improvement += 1
                print('Epochs since last improvement: {epochs_since_improvement}')
                best_score = valid_score
            else:
                epochs_since_improvement = 0

                state_dict = {'epoch': epoch,
                              'epochs_since_improvement': epochs_since_improvement,
                              'decoder': decoder,
                              'decoder_optimizer': decoder_optimizer,
                              'encoder': encoder,
                              'encoder_optimizer': encoder_optimizer,
                              'valid_score': valid_score,
                              'best_score': best_score}

                filename = 'checkpoint.pth.tar'
                torch.save(state_dict, filename)


            # Print log info
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                print(
                    'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f}), Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                        top1=top1, top5=top5))
                print('Validate score %.3f'%(valid_score))


def validate(data_loader_valid, encoder, decoder, criterion, vocab):
    encoder.eval()
    decoder.eval()
    valid_losses = AverageMeter()

    alpha_c = config.get('alpha_c', 0.)

    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(data_loader_valid):
            images = images.to(device)
            captions = captions.to(device)

            features = encoder.forward(images)
            prediction, alphas = decoder.forward(features, captions)

            att_regularization = alpha_c * ((1 - alphas.sum(1)) ** 2).mean()
            loss = criterion(prediction.permute(0, 2, 1), captions) + att_regularization

            total_caption_length = calculate_caption_lengths(vocab.word2idx, captions)
            valid_losses.update(loss.item(), total_caption_length)

    return valid_losses.avg
