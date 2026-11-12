# Lab 1 Notes – Fundamentals of Machine Learning and Systematic Experimentation

**Course:** D7041E Applied Artificial Intelligence
**Sources:** `Lab1-Fundamentals.pdf` (6 pages) and the starter notebook `D7041E_code_for_lab1.ipynb`. The data file `vecs.npy` belongs to Task 1.2.

**What this lab makes you do:**

1. Build your own small image dataset and preprocess it.
2. Load someone else's awkwardly packaged dataset.
3. Run, fix and extend a nearest-neighbour classifier on MNIST.
4. Pick the best k with cross-validation.
5. Train SVMs with three kernels on the Iris dataset, in two multi-class styles.
6. Run a proper experiment campaign on 10 UCI datasets and write it up.

---

## Rules you must not break

- **It is strictly forbidden to use AI-generated code or to copy code from the internet. Violations are reported.**
- You hand in a report with answers to all questions and screenshots of intermediate steps.
- **You pass only by explaining the lab verbally to a TA.** The report alone is not enough. The TA will ask *you*, individually.
- Deadline: end of Friday of the practice week (from Lecture 1).

---

## Part 1. Warming up: data handling

### Task 1.1 – Your own dataset

Take **at least 5 photos of each of 5 categories** of objects (25+ photos).

**1. Put everything in one array of shape $(n, x, y, 3)$.**
- $n$ = number of photos.
- $x$ and $y$ = the photo's height and width in pixels.
- $3$ = the red, green and blue channels.

Worked example: 25 photos, each $200 \times 300$ pixels, colour. Shape $= (25, 200, 300, 3)$. Number of stored values:

$$
n \cdot x \cdot y \cdot 3 = 25 \times 200 \times 300 \times 3 = 4{,}500{,}000 .
$$

Step by step: pixels per photo $200 \times 300 = 60{,}000$; times 3 channels $= 180{,}000$; times 25 photos $= 4{,}500{,}000$ numbers. Big, which is one reason preprocessing matters.

**2. Write PlotSample(index).** A function that takes a sample number and shows that photo with matplotlib. The notebook already shows the pattern for MNIST: pick the image, set a title with its label, call imshow, show, close.

**3. Follow the "Preprocessing for deep learning" tutorial** (linked from Canvas) and apply every step to your dataset. You must be able to demonstrate the notebook with all steps. Typical steps: scale pixel values to a common range, subtract the mean, and so on. Remember from Lecture 2: fit preprocessing settings on the training part only, then apply them to the test part, to avoid data leakage.

### Task 1.2 – Somebody else's dataset

The story: a collaborator sends 11,000 samples. Each sample is a **1024-dimensional embedding** (a feature vector of 1024 numbers produced by some other model) of a handwritten digit placed at one of **9 positions** on a scene. The file is `vecs.npy`. It is packaged as **a dictionary inside a dictionary**: the outer key is the **position**, the inner key is the **digit**, and under that sit the vectors. It took the lecturer about an hour to get it into NumPy arrays ready for an algorithm. How long does it take you?

**What to deliver:**
1. Code that, for **any chosen position**, automatically builds **two NumPy arrays**: one with the embeddings and one with the labels (the digits), in the same order as in the file.
2. Code that **randomly permutes** the data and the labels **together** (same shuffle for both, or the labels stop matching their rows).

In plain words: the outer dictionary is a filing cabinet with 9 drawers (positions). Each drawer has 10 folders (digits 0–9). Each folder holds a stack of 1024-number vectors. Your code opens one drawer, goes through every folder, stacks all the vectors into one array and writes the folder's digit into the label array for each one.

---

## Part 2. Nearest-neighbour classifier

The lab is honest: this method "has little to do with learning". Its job here is to teach you dataset handling and to be a **baseline**, a simple reference score that later methods must beat.

### The MNIST dataset

Tiny 28 × 28 pixel images of handwritten digits, 10 classes ("0" to "9"). The lab text says 60,000 images split 50,000 train / 10,000 test.

> **Note:** the standard MNIST download (and the `mnist.load_data()` call in the notebook) gives 60,000 training images plus 10,000 test images. The notebook then uses only the first 500 training images to keep things fast. Check which numbers you actually have before you report them.

### What the notebook does

1. Loads MNIST into training images/labels and test images/labels.
2. Shows one or two images with their labels.
3. Takes the first 500 training images and **reshapes** each $28 \times 28$ image into one row of $28 \times 28 = \mathbf{784}$ numbers. That turns a picture into a feature vector (Lecture 2, section 1).
4. Defines `predict(X)`: for each test image, subtract it from every training row, take absolute values, **sum along each row**, find the row with the smallest sum, and copy its label. That sum is the **$L_1$ distance**:
   $$
   L_1(\text{test}, \text{train}_j) = \sum_{p=1}^{784} \left| \text{test}_p - \text{train}_{j,p} \right|
   $$
   and the prediction is the label of $\arg\min_j L_1(\text{test}, \text{train}_j)$.
5. Reshapes the test images the same way, predicts, and prints accuracy as the share of predicted labels that equal the true labels.

### Task 2.1

1. **Run the 1-NN code. What accuracy do you get?**
2. **Change $L_1$ to $L_2$.** $L_2$ = square root of the sum of squared differences (Lecture 2, section 5):
   $$
   L_2(\text{test}, \text{train}_j) = \sqrt{ \sum_{p=1}^{784} \left( \text{test}_p - \text{train}_{j,p} \right)^2 }
   $$
   Square instead of absolute value, sum, square root. How does accuracy change?
3. **The accuracy is VERY low because of a bug. Find it and fix it.** The hint: compute the distance between a few pairs of images by hand and compare with what Python gives.
   *A pointer (my addition, consistent with the hint):* MNIST pixels are stored as unsigned 8-bit integers, values 0 to 255, with no negative numbers possible. Subtracting a bigger pixel from a smaller one "wraps around": $3 - 5$ becomes $254$ instead of $-2$ (because $3 - 5 + 256 = 254$). So the distances are garbage. Think about what number type the arrays should have *before* you subtract. Verify by hand as the lab asks; the TA will ask you to explain it.
4. **Extend to k-NN.** Instead of the single closest image, find the **k** closest and let them **vote** on the label. You need: sort the distances, take the first k indices, look up their labels, pick the most common.

### Task 2.2 – Hyperparameters and cross-validation

k-NN has settings: the value of **k** and the choice of **distance** (L1 or L2). These are **hyperparameters** (Lecture 2, section 9). The lab says in bold:

> **In machine learning one should avoid adjusting these parameters based on the test data!**

If you tune on the test set, you've tuned to that set, and your score is flattering. Use a separate **validation** set, or when data is small, **cross-validation**.

**The figure in the lab:** the training data is cut into fold 1 … fold 5, and the test data sits apart in red. One fold (the yellow fifth one) is used for validation while the other four train. Then you rotate which fold validates. Typical: 3, 5 or 10 folds.

> **In the very end, once the model is trained and the best hyperparameters are found, the model is evaluated a single time on the test data.**

**Your tasks:**
1. **Find the best k with 3-fold cross-validation, using L2.**
   - Split the training data into folds A, B, C.
   - For each candidate k: train on B+C and validate on A; train on A+C and validate on B; train on A+B and validate on C. Average the three accuracies.
   - Pick the k with the best average.
2. **Report the accuracy on the test data for that best k.** One run, one number.

---

## Part 3. Support Vector Machines

SVM is "one of the most widely used algorithms for classification". It brings in model-based learning, high-dimensional spaces and separability (Lecture 3). Here you use scikit-learn's implementation.

### Task 3.1 – Iris

1. Load the **Iris** dataset (150 flowers, 4 features, 3 species).
2. Split **80:20** into training and test.
3. Train SVMs with **linear**, **polynomial** and **RBF** kernels for multi-class classification. A kernel is the similarity function that replaces the dot product in the SVM (Lecture 3, section 7). Linear = plain straight boundaries; polynomial = curved boundaries whose degree sets how wiggly; RBF (radial basis function) = very flexible, can follow almost any smooth boundary.

**Deliver:**
1. **Confusion matrices** for **one-vs-one** and **one-vs-rest** training, for each kernel. (Lecture 3, section 9: one-vs-one builds $\frac{m(m-1)}{2} = \frac{3 \cdot 2}{2} = 3$ machines for 3 classes; one-vs-rest builds $m = 3$, one per class. Lecture 2, section 8: how to read the matrix.)
2. **Which kernel gives the best accuracy and $F_1$ score?** (Lecture 2, section 7 for the formulas: $\text{Accuracy} = \frac{TP+TN}{\text{all}}$, $F_1 = \frac{2PR}{P+R}$.)
3. **Extract the support vectors for each class** in the one-vs-rest case. These are the training points with $\alpha_i > 0$ that define each boundary.
4. **Plot the decision boundary** for features 2 vs 3 and 3 vs 4.

---

## Part 4. Larger-scale systematic experimentation

Read the paper by Delgado et al., *Do we need hundreds of classifiers…*, JMLR 2014 (the link is in the lab PDF). It benchmarks many classifiers on **121 UCI datasets**. From Lecture 2: 19 of those come as separate train and test files, 102 as a single file you must split.

**Your job:**
- Choose **10** of the 121 datasets.
- Develop and evaluate **two unsupervised and two supervised** classification models from scikit-learn.
- Compare with the baseline results reported in the paper.
- Show **appropriate preprocessing**, and **systematic experimentation**: correct transfer of parameters between train, validation and test sets (fit scalers on train only), **random seeds**, and **averaging** results over seeds.

**The report must contain:**
1. Description of the datasets.
2. Description of the models.
3. Description of the methodology: splits, cross-validation, number of experiments with random seeds, performance metrics.
4. Experimental results.
5. Conclusions.

This is exactly the "Important" page of Lecture 2: automate your experiments, save results together with their configuration, so everything can be repeated.

---

## How the lab maps to the lectures

| Lab task | Lecture idea |
|---|---|
| (n, x, y, 3) arrays, flattening to 784 | Feature vectors (L2 §1) |
| Two arrays: data + labels | Dataset and labels (L2 §3) |
| 1-NN, k-NN voting | Memory-based learning (L2 §4) |
| L1 vs L2 | Distances (L2 §5) |
| Confusion matrices, accuracy, F1 | Assessing a classifier (L2 §7–8) |
| k and distance as hyperparameters | Parameters vs hyperparameters (L2 §9) |
| 3-fold CV, test once | Splits and cross-validation (L2 §10) |
| Fit preprocessing on train only, seeds, averaging | Leakage and seeds (L2 §10) |
| Kernels, support vectors, one-vs-one / one-vs-rest | SVM (L3 §7–9) |
| Why high dimensions are hard | Curse of dimensionality (L1 §5) |

## Key points to remember

- No AI code, no copied code. Verbal, individual assessment.
- Images → arrays of shape (n, x, y, 3); each image flattens to one feature vector.
- The provided 1-NN uses L1; switch to L2; find the distance bug by hand-checking a few pairs; extend to k-NN voting.
- Never tune on the test set. 3-fold CV to pick k; test once.
- Iris with linear, polynomial, RBF kernels; one-vs-one and one-vs-rest; confusion matrices, accuracy, F1, support vectors, boundary plots.
- Ten UCI datasets, two supervised and two unsupervised models, preprocessing fitted on train only, several seeds, averaged, all written up.

## Key terms

- **Baseline** – a simple method you compare better ones against.
- **Embedding** – a feature vector produced by another model (here 1024 numbers per digit).
- **Flattening / reshaping** – turning a 28 × 28 image into a 784-long row.
- **L1 / L2 norm** – Manhattan / Euclidean distance.
- **Hyperparameter** – a setting you choose (k, distance, kernel).
- **Cross-validation (k-fold)** – rotate the validation fold over the training data, average.
- **Kernel (linear, polynomial, RBF)** – the similarity function inside an SVM.
- **One-vs-one / one-vs-rest** – pairwise SVMs / one SVM per class.
- **Support vectors** – the training points that define the SVM boundary.
- **Data leakage** – test information sneaking into training.
- **Seed** – fixes the random number sequence so runs repeat.

## Questions a TA might ask you

**1. Why is the 1-NN accuracy so low at first?**
Because the distances are computed wrongly. Check: the pixel arrays are unsigned 8-bit integers, so subtraction wraps around instead of going negative. Fix the data type before subtracting, and verify against a hand calculation.

**2. Why 3-fold cross-validation instead of just using the test set to pick k?**
Picking k on the test set tunes to that set and inflates the reported score. Cross-validation uses only training data to choose k; the test set is touched once at the end.

**3. How many SVMs does one-vs-one train for Iris? And one-vs-rest?**
Three classes: $\frac{3 \times 2}{2} = 3$ for one-vs-one; $3$ for one-vs-rest.

**4. Why fit the scaler on the training set only?**
Otherwise the mean and spread of the test data influence training: data leakage.

**5. Why repeat experiments with different seeds?**
Random initialisation makes single runs lucky or unlucky; averaging over seeds gives a fair, repeatable result.

**6. What is a support vector, and how do you get them out of a trained one-vs-rest model?**
A training point sitting on the margin edge with $\alpha_i > 0$. scikit-learn exposes them on the fitted model; in one-vs-rest you get the set for each class's machine.
