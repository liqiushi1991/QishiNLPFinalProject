# All configurations
import torchvision.models as models

""" Config various (hyper)parameters. """
# Model architecture
config = {
    # General
    'image_net': models.resnet101,

    # Optimizer
    'log_step': 10,
    'num_epochs': 5,
    'decoder_hidden_size': 512,
    'batch_size': 32,
    'clip_gradients': 5.0,
    'alpha_c': 1,
    'crop_size': 256,
    'epochs_since_improvement': 0,
    'encoder_lr': 1e-4,
    'decoder_lr': 4e-4,
    'checkpoint': None,

    # Data set locations
    'train_caption_path': 'data/annotations/captions_val2017.json',
    'validate_caption_path': 'data/annotations/captions_val2017.json',
    'test_caption_path': 'data/annotations/captions_val2017.json',

    'train_image_dir': 'data/val2017/',
    'validate_image_dir': 'data/val2017/',
    'test_image_dir': 'data/val2017/',
    'model_path': 'model/',

    # Build Vocabulary
    'vocab_path': 'data/vocab.pkl',  # Path to save vocabulary
    'threshold': 4,  # Minimum word count threshold
    'overwrite': False,  # Regenerating vocabulary or not

    # Resize Image
    'train_resized_image_dir': 'data/resizedval2017/',
    'validate_resized_image_dir': 'data/resizedval2017/',
    'test_resized_image_dir': 'data/resizedval2017/',
    'image_size': (256, 256),

    # Data loader
    'workers': 1,
    'img_mean': (0.485, 0.456, 0.406),
    'img_std': (0.229, 0.224, 0.225),
    'shuffle': True,

}
