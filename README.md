### MNIST-CLASSIFICATION-TUTORIAL
#### Overview
- Algorithms
    - Machine learning: LR, SVM, XGBoost, FFNN
    - Deep learning: CNN, ResNet, Distilling Knowledge, Data-Free Learning 
- Framework
    - Sklearn
    - Tensorflow
    - Pytorch
#### Progress
Model | Main Params | Test Accuracy | Time Cost (s) | Comments
---| --- | --- | --- | ---
LR | solver='liblinear', multi_class='ovr' | 0.9202 | 58.65
SVM | kernel='rbf', decision_function_shape='ovr' | 0.9446 | 552.20
SVM | kernel='rbf', decision_function_shape='ovr' | 0.8535 | 35.67 | count white dots per row as features
XGBoost | max_depth=5, n_jobs=10 | 0.9651 | 149.56
XGBoost | max_depth=5, n_jobs=10 | 0.8461 | 15.02 | count white dots per row as features
FFNN |
CNN |
ResNet |
Distilling Knowledge |
Data-Free Learning |

#### Reference
- [THE MNIST DATABASE
of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [ConvNetJS MNIST demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)