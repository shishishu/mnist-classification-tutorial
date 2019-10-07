### MNIST-CLASSIFICATION-TUTORIAL
#### Overview
- Algorithms
    - Machine learning: LR, SVM, XGBoost, MLP
    - Deep learning: CNN, ResNet, VAE, Distilling Knowledge, Data-Free Learning 
- Framework
    - Sklearn
    - Tensorflow
    - Pytorch
#### Progress
Model | Framework | Main Params | Test Accuracy | Time Cost /s | Comments
---| --- | --- | --- | --- | ---
LR | sklearn | solver='liblinear', multi_class='ovr' | 0.9202 | 57.87
SVM | sklearn | kernel='rbf', decision_function_shape='ovr' | 0.9446 | 556.91
XGBoost | sklearn | max_depth=5, n_jobs=10 | 0.9651 | 141.38
MLP | sklearn | hidden_layer_sizes=(128, 32) | 0.9768 | 44.80
MLP | tensorflow | batch_size=512, learning_rate=1e-3, hidden_layers=[128,32]| 0.9725 | 43.84
CNN | tensorflow | batch_size=256, learning_rate=1e-5, num_epoch=200 | 0.9704 | 2181.92
ResNet | 
VAE | 
Distilling Knowledge |
Data-Free Learning |

#### Reference
- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [ConvNetJS MNIST demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)
- [Feed-Forward Neural Net for MNIST](https://wpovell.net/posts/ffnn-mnist.html)