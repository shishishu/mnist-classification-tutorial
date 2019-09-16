### MNIST-CLASSIFICATION-TUTORIAL
#### Overview
- Algorithms
    - Machine learning: LR, SVM, XGBoost, MLP
    - Deep learning: CNN, ResNet, Distilling Knowledge, Data-Free Learning 
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
MLP | sklearn | hidden_layer_sizes=(128, 32), activation='relu' | 0.9811 | 44.80
CNN |
ResNet |
Distilling Knowledge |
Data-Free Learning |

#### Reference
- [THE MNIST DATABASE
of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [ConvNetJS MNIST demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)