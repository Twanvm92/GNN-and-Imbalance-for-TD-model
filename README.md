# GNN-and-Imbalance-for-TD-model

Self-Admitted Technical Debt Detection by Learning Its Comprehensive Semantics via Graph Neural Networks

## 1. Environment Setup

Python library dependencies:

- tensorflow -v : 1.14.0
- numpy -v : 1.19.3
- gensim -v : 3.8.1
- scipy -v : 1.4.1
- others: sklearn

## 2. Data Preprocessing

Dataset:

Everton da S. Maldonado, Emad Shihab, Nikolaos Tsantalis: Using Natural Language Processing to Automatically Detect Self-Admitted Technical Debt. IEEE Trans. Software Eng. 43(11): 1044-1062 (2017)

**If you want to use our model quickly, you can use the test.py file directly and modify the path to the model in the file to get our results quickly. **

## 3. Graph Preparation

- remove_words.py

  ```python
  python remove_words.py 
  ```

If you want to replace the data set, you can use the variable dataset = 'dataset name' in the remove_words.py file. 

- build_graph.py

  ```python
  python build_graph.py
  ```

If you want to replace the data set, you can use the variable dataset = 'dataset name' in the build_graph.py file. 

## 4. Train

- train.py

  ```python
  python train.py
  ```

If you want to replace the data set, you can use the variable dataset = 'dataset name' in the train.py file. There are also a number of parameters that can be modified to suit your needs, such as learning rate, epochs, etc.

## 5. Test

- test.py

  ```
  python test.py
  ```

If you want to replace the data set, you can use the variable dataset = 'dataset name' in the test.py file. 
