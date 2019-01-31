# Model Architecture:
# Convolution: ['C', in_channels, out_channels, (kernel), stride, dilation, padding, Activation Function]
# Max Pooling: ['M', (kernel), stride, padding]
# Average Pooling: ['A', (kernel), stride, padding]
# Linear Layer: ['L', in_features, out_features, Activation Function]
# Dropout : ['D', probability]
# Dropout 2D : ['D2d', propability]
# Alpha Dropout : ['AD', probability]
# Classifying layer: ['FC', in_features, num_classes]
# Possible Activation Fns: 'ReLU', 'PReLU', 'SELU', 'LeakyReLU', 'None'->(Contains no Batch Norm for dimensionality reduction 1x1 kernels)
# srun python main.py --batch-size 16 --epochs 50 --lr 0.001 --momentum .9 --log-interval 100 --root-dir ../ --train-input-file ../clipped_training_data --train-target-file ../clipped_training_targets --test-input-file ../clipped_test_data --test-target-file ../clipped_test_targets

# The calculations below are constrained to stride of 1
# padding of 2 for 3x3 dilated convolution of 2 for same input/output image size
# padding of 3 for 3x3 dilated convolution of 3
#
# padding of 4 for 5x5 dilated convolution of 2 for same input/output image size
# padding of 6 for 5x5 dilated convolution of 3
#
# padding of 6 for 7x7 dilated convolution of 2 for same input/output image size
# padding of 9 for 7x7 dilated convolution of 3


feature_layers = {

    '1': [['C', 3, 16, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 16, 24, (3,3), 1, 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 24, 36, (3,3), 1, 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0]],

    '2': [['C', 3, 16, (3,3), 1, 1, 1, 'ReLU'], ['C', 16, 24, (3,3), 1, 1, 1, 'ReLU'], ['C', 24, 36, (3,3), 1, 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 36, 8, (1,1), 1, 1, 0, 'ReLU'], ['C', 8, 16, (3,3), 1, 1, 1, 'ReLU'], ['C', 16, 24, (3,3), 1, 1, 1, 'ReLU'], ['C', 24, 36, (3,3), 1, 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 36, 8, (1,1), 1, 1, 0, 'ReLU'], ['C', 8, 16, (3,3), 1, 1, 1, 'ReLU'], ['C', 16, 24, (3,3), 1, 1, 1, 'ReLU'], ['C', 24, 36, (3,3), 1, 1, 1, 'ReLU'],
    ['M', (15,15), 1, 0]],
}

classifier_layers = {
    '1': [['L', 36 * 15 * 15, 368, 'ReLU'], ['D', .5], ['FC', 368, 12]],
    '2': [['L', 36 * 1 * 1, 512, 'ReLU'], ['D', .2], ['FC', 512, 12]],
}


