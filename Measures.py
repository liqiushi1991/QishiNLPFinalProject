class AverageMeter(object):
    '''From https://github.com/pytorch/examples/blob/master/imagenet/main.py'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    pred = pred.permute(1, 0, 2)
    correct = pred.eq(targets.expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    print(correct_total, correct.numel())
    return correct_total.item() / float(correct.numel() / k)


def calculate_caption_lengths(word_dict, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<end>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths
