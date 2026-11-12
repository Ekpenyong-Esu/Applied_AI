# Lab 1

```
Lab1/
  Lab1_Solution.ipynb        my solution, run this
  D7041E_code_for_lab1.ipynb the lecturer's original notebook
  lab1/
    config.py                seed, paths, photo size
    image_dataset.py         Task 1.1   photos -> (n, x, y, 3) array
    preprocessing.py         Task 1.1   covariance, center, standardize, decorrelate, whiten, ZCA
    multidigit_mnist.py      Task 1.2   vecs.npy -> embeddings + labels for one position, shuffle
    mnist_data.py            Part 2     load + flatten MNIST (from mnist.npz)
    distances.py             Part 2     L1 / L2 and the uint8 bug
    nearest_neighbour.py     Part 2     1-NN / k-NN
    cross_validation.py      Part 2     3-fold CV to choose k
    metrics.py               Parts 2-4  accuracy, confusion matrix, F1
    plotting.py              all        figures, saved to figures/
    svm_iris.py              Part 3     SVM kernels, one-vs-one / one-vs-rest, support vectors
    uci_experiments.py       Part 4     10 UCI datasets x 4 models x 5 seeds
  my_dataset/<category>/*.jpg   photos for Task 1.1
  vecs.npy                   collaborator's data for Task 1.2
```

## Running
1. `pip install -r requirements.txt` (the pandas installed on this machine is too old and breaks sklearn)
2. Put the photos in `my_dataset/<category_name>/`.
3. Start the kernel from the `Lab1` folder so `import lab1` works.
