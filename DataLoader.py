import torch
import os
import nltk

import torch.utils.data  as data
from pycocotools.coco    import COCO
from PIL                 import Image
from BuildVocab          import build_and_save_vocab, Vocabulary
from torchvision         import transforms
from ResizeImage         import resize_images
from Config              import config
from utils               import absolute_dir_wrapper


class CocoDataset(data.Dataset):
    """Customized dataset to use torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """
        Args:
            root: image dir.
            json: coco captions dir.
            vocab: vocabulary.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # Convert caption to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, resized_image_size, resized_image_size).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, resized_image_size, resized_image_size).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length, descending
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images
    images = torch.stack(images, 0)

    # Merge captions
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    # Padding
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths


def get_loader(data_type):
    if data_type not in ('train', 'test', 'validate'):
        raise ValueError("data_type has to be 'train', 'test' or 'validate'.")

    resize_images(data_type)

    root = absolute_dir_wrapper(config.get('%s_resized_image_dir' % data_type))
    json = absolute_dir_wrapper(config.get('%s_caption_path' % data_type))
    vocab = build_and_save_vocab()

    transform = transforms.Compose([
        transforms.RandomCrop(config.get('crop_size')),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.get('img_mean'),
                             config.get('img_std'))])

    batch_size = config.get('batch_size')
    shuffle = config.get('shuffle')
    num_workers = config.get('workers')

    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
