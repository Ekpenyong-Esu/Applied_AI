# Quiz 1 Notes – Every Question Explained, Plus What to Revise

**Course:** D7041E Applied Artificial Intelligence
**Source:** `quiz1.txt`, a Canvas results export of Quiz 1 (14 multiple-answer questions). The export shows, for each option, whether it was a correct answer, a distractor, or something you got wrong.

**How to read this file.** For each question you get: the correct options as the quiz grades them, what the wrong options were trying to trick you with, and a plain-language explanation tied to the lecture notes. At the end: your score breakdown and a short revision list.

> Note on the export: the first question's score line is cut off at the top of the file, so its points are not shown. Three questions (2, 8, 13) were re-graded upward, which usually means the teacher accepted all four options as correct after students complained.

---

## Your score at a glance

| Q | Topic | Your points | Comment |
|---|---|---|---|
| 1 | Kernel comparisons | not shown | all four options correct |
| 2 | SVM multi-class strategies | 4 / 4 (regraded from 2.667) | all four correct |
| 3 | Choice of distance in k-NN | 4 / 4 | all four correct |
| 4 | Distances for x = [1,2,3], y = [4,6,3] | 4 / 4 | A and B correct |
| 5 | Normalising test data separately | **1.5 / 3** | you picked A but missed B |
| 6 | Cross-validation objectives | 4 / 4 | A, B, C correct |
| 7 | Kernel trick enables SVM to | **2.667 / 4** | you picked D, which was wrong |
| 8 | SVM decision margin | 3 / 3 (regraded from 2) | all four correct |
| 9 | 10-fold vs 3-fold CV | 4 / 4 | A, B, C correct |
| 10 | Supervised vs unsupervised | 3 / 3 | A, B correct |
| 11 | Scaling techniques | 4 / 4 | A, B correct |
| 12 | Overfitting in high dimensions | 3 / 3 | A, B correct |
| 13 | k-NN properties | 3 / 3 (regraded from 2) | all four correct |
| 14 | Feature scaling without normalisation | 3 / 3 | A, B, C correct |

Points shown: 43.17 out of 46, plus question 1. You lost points in exactly two places, both explained below.

---

## Question 1 – Kernel comparisons

**Correct (all four):**
- A. RBF kernel can approximate any smooth boundary.
- B. Linear kernel is computationally cheap.
- C. Polynomial degree controls nonlinearity.
- D. Kernel choice affects accuracy and training time.

**Why.** A kernel is the similarity function that replaces the dot product inside an SVM (Lecture 3, section 7). Think of it as choosing what kind of pen you draw the boundary with.
- **Linear kernel** = a ruler. Straight boundaries only, and very fast because it's just a dot product.
- **Polynomial kernel** = a bendy ruler. The **degree** (2, 3, 4 …) says how bendy. Degree 1 is just linear; higher degrees allow more curves.
- **RBF kernel** (radial basis function) = freehand. With the right settings it can follow almost any smooth shape. Flexible, but slower and easier to overfit.
- Because the kernels differ this much, picking one changes both how accurate you are and how long training takes.

---

## Question 2 – SVM multi-class strategies

**Correct (all four, after regrade):**
- A. One-vs-One trains n(n−1)/2 classifiers.
- B. One-vs-Rest trains one classifier per class.
- C. Results are not identical for both methods.
- D. Each classifier outputs a score/confidence.

**Why.** An SVM separates two classes, so for $n$ classes you need a strategy (Lecture 3, section 9).
- **One-vs-one:** one machine per pair. $n$ classes pair up in
  $$
  \frac{n(n-1)}{2}
  $$
  ways (each class meets $n-1$ others; halve it because "cat vs dog" is the same as "dog vs cat"). For 3 classes that's $\frac{3 \cdot 2}{2} = 3$; for 10 classes $\frac{10 \cdot 9}{2} = 45$.
- **One-vs-rest:** one machine per class, each asking "is it this class or anything else?". $n$ machines.
- Each machine gives a **score** (how far the sample is from its boundary, with a sign). One-vs-rest picks the class with the biggest score; one-vs-one counts votes.
- They train different machines on different subsets, so they can disagree. **Not identical.**

---

## Question 3 – Choice of distance in k-NN

**Correct (all four):**
- A. L1 norm is more robust to outliers.
- B. Higher-order norms (L2 … Lp) emphasise larger deviations.
- C. Cosine distance ignores vector magnitude.
- D. Mahalanobis distance uses feature correlations.

**Why.** Lecture 2, sections 5–6.
- **$L_1$** adds absolute differences, $\sum_i |\Delta_i|$. One huge difference adds just itself.
- **$L_2$** squares the differences first, $\sqrt{\sum_i \Delta_i^2}$. A difference of 10 becomes 100; a difference of 1 becomes 1. Big deviations get amplified. Higher $p$ in the Minkowski distance $\left(\sum_i |\Delta_i|^p\right)^{1/p}$ amplifies even more. So $L_2$ and above **emphasise large deviations**, and $L_1$, which doesn't, is **more robust to outliers**.
- **Cosine** divides by the vectors' lengths, so doubling a vector changes nothing. It **ignores magnitude** and looks at direction (the shopping-basket example).
- **Mahalanobis distance** (not in the lectures, so here is the idea): it first stretches and rotates the space so that every feature has the same spread and correlated features don't get counted twice, then measures Euclidean distance there. Picture a cigar-shaped cloud of points: Mahalanobis says "far" for a point off the cigar's side even if it's close in metres, because sideways spread is rare. That's what "uses feature correlations" means.

---

## Question 4 – Distances for $x = [1, 2, 3]$, $y = [4, 6, 3]$

**Correct:** A. $L_1$ distance $= 7$. B. $L_2$ distance $= \sqrt{25} = 5$.
**Distractors:** C. $L_2$ distance $= \sqrt{14}$. D. Cosine similarity $\approx 0.94$.

**Work it out (Lecture 2, section 5):**

1. Differences: $1 - 4 = -3$, $2 - 6 = -4$, $3 - 3 = 0$.
2. Absolute values: $3, 4, 0$.
3. $$L_1 = |{-3}| + |{-4}| + |0| = 3 + 4 + 0 = \mathbf{7}$$ ✓ A.
4. Squares: $9, 16, 0$. Sum $= 25$.
   $$L_2 = \sqrt{(-3)^2 + (-4)^2 + 0^2} = \sqrt{25} = \mathbf{5}$$ ✓ B.

**Why C is a trap.** $\sqrt{14} = \sqrt{1^2 + 2^2 + 3^2} = \lVert x \rVert$ is the **length of $x$ itself**, not the distance between $x$ and $y$. Classic bait.

**Why D is wrong.** $\cos = \dfrac{x \cdot y}{\lVert x \rVert \, \lVert y \rVert}$.

1. Dot: $1 \times 4 + 2 \times 6 + 3 \times 3 = 4 + 12 + 9 = 25$.
2. $\lVert x \rVert = \sqrt{14} \approx 3.742$. $\lVert y \rVert = \sqrt{16 + 36 + 9} = \sqrt{61} \approx 7.810$.
3. $$\cos = \frac{25}{3.742 \times 7.810} = \frac{25}{29.22} \approx \mathbf{0.856},$$ not $0.94$.

---

## Question 5 – Normalising test data separately  (you scored 1.5 / 3)

**The question:** if test data are normalised independently from training data, then …

**Correct:** A. Inconsistent feature scaling may occur. **B. Accuracy can be artificially inflated.** (You missed B.)
**Distractors:** C. Data leakage from test to train is introduced. D. No impact on results.

**What "normalising separately" means.** You compute the mean and spread of the *training* set and scale the training features with them. Then you compute a *different* mean and spread from the *test* set and scale the test features with those.

**Why A is correct.** The two sets are now on different rulers. A value of 0.5 in training doesn't mean the same thing as 0.5 in test. The model was trained on one ruler and is asked about another. That's **inconsistent scaling** (Lecture 2, section 10, "do configuration of data preprocessing separately for train & test sets" means: fit on train, *apply* to test, not re-fit on test).

**Why B is also correct (the one you missed).** When you re-fit on the test set you are using information about the test set (its own statistics) to shape the test inputs. That is a form of peeking: the test set is no longer a clean, untouched sample. Results from such a setup are not trustworthy and **can look better than they really are**. The quiz treats this as "accuracy can be artificially inflated". (It can also be deflated; the point is that the number is no longer honest.)

**Why C is wrong (a subtle one).** Leakage "from test to train" happens when test data influence *training*, for example when you fit the scaler on train + test together. Normalising the test set *on its own* does not feed anything into training. So the direction in C is wrong.

**Why D is wrong.** Obviously there is an impact.

**The right procedure, once and for all:**
1. Fit the scaler on the training data only (compute mean and spread there).
2. Apply those same numbers to training, validation and test data.
3. Never compute statistics from the test set.

---

## Question 6 – Cross-validation objectives

**Correct:** A. Estimate generalisation performance. B. Tune hyperparameters without using the test set. C. Avoid using test data for model selection.
**Distractor:** D. Optimise learning rate.

**Why.** Lecture 2, section 10. Cross-validation rotates the validation fold through the training data and averages. That gives (A) a fair estimate of how the model will do on unseen data, and lets you (B, C) choose settings and models while the test set stays sealed.

**Why D is wrong.** The learning rate is *one* hyperparameter, and k-NN doesn't even have one. Cross-validation is a general tool, not specifically a learning-rate optimiser. The option is too narrow to be a stated "objective".

---

## Question 7 – Kernel trick enables SVM to …  (you scored 2.667 / 4)

**Correct:** A. Compute dot products in high-dimensional space efficiently. B. Represent non-linear decision boundaries. C. Avoid explicit feature construction.
**Wrong, and you selected it:** D. Does not always reduce dimensionality.

**Why A, B, C.** Lecture 3, sections 7 and 10. In the dual problem the data only appear as dot products $\mathbf{x}_i \cdot \mathbf{x}_j$. A kernel $K(\mathbf{x}_i, \mathbf{x}_j)$ returns what the dot product *would be* after mapping the points into a higher-dimensional space, without ever building that space (A: efficient; C: no explicit new features). In that higher space a straight cut corresponds to a **curved** boundary in the original space (B). The circle-inside-a-ring example: adding $z = x_1^2 + x_2^2$ as a new axis makes the classes separable with a flat plane.

**Why D is wrong.** The kernel trick has nothing to do with *reducing* dimensions. It secretly goes **up** in dimension (sometimes to infinitely many, with RBF). "Does not always reduce dimensionality" is a muddled statement about the wrong direction; it is not something the trick "enables". Dimensionality *reduction* is PCA, ICA, t-SNE (Lecture 2, section 2), a different family of tools.

**Memory hook.** Kernel trick = lift up. PCA = squash down.

---

## Question 8 – SVM decision margin

**Correct (all four, after regrade):**
- A. Maximises margin between classes.
- B. Not limited to linear separability.
- C. Kernels enable non-linear boundaries.
- D. Sensitive to feature scaling.

**Why.** Lecture 3.
- A: the whole objective is the widest empty strip, $D = \dfrac{2}{\lVert \mathbf{w} \rVert}$, achieved by minimising $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2$.
- B and C: with a kernel, the SVM handles data no straight line can split.
- D: the margin is measured as a distance. Distances depend on the scale of features (Lecture 2, section 6). A feature in thousands dominates one in decimals, so the margin gets pulled around by whichever feature has the biggest numbers. Scale your features before training an SVM.

---

## Question 9 – 10-fold vs 3-fold cross-validation

**Correct:** A. 10-fold gives lower-variance estimates. B. Increases computational cost. C. Uses more data for training per fold.
**Distractor:** D. It does not overfit by design.

**Why.** With $n$ folds you train on $\frac{n-1}{n}$ of the training data each time.
- 3-fold: train on $\frac{2}{3} \approx 67\%$ per round, 3 rounds.
- 10-fold: train on $\frac{9}{10} = 90\%$ per round, 10 rounds.
So 10-fold trains on **more data per fold** (C), costs **more** because you fit 10 models instead of 3 (B), and the quiz takes the usual textbook position that averaging over 10 rounds gives a **steadier (lower-variance)** estimate (A).

**Why D is wrong.** Cross-validation is a way of *measuring*; it doesn't stop a model from overfitting. A too-complex model overfits inside each fold just the same. There is no "by design" protection.

---

## Question 10 – Supervised vs unsupervised learning

**Correct:** A. Supervised learning requires labelled data. B. Unsupervised learning groups data without labels.
**Distractors:** C. Both can perform classification tasks. D. Unsupervised learning optimises label prediction.

**Why.** Lecture 1, section 4.
- Supervised = the answers (labels) are given. Needs labelled data (A).
- Unsupervised = no answers; the typical task is clustering, grouping without labels (B).
- Classification means predicting a *given* label set, which requires labels, so it's a supervised task. Clustering looks similar but is not classification; the groups have no names (C wrong). Unsupervised learning has no labels to optimise toward (D wrong).

---

## Question 11 – Scaling techniques for distance-based algorithms

**Correct:** A. Z-score (standard) normalisation. B. Min-Max normalisation.
**Distractors:** C. Logarithmic normalisation. D. Rank normalisation.

**Why.** k-NN and SVM both work with distances, and distances are scale-sensitive (Lecture 2, section 6; Q8 D above). The two standard fixes:

- **Z-score:**
  $$
  x' = \frac{x - \mu}{\sigma}
  $$
  where $\mu$ is the mean and $\sigma$ the standard deviation (computed on the training set). Every feature ends up centred at 0 with spread 1. Example: heights with $\mu = 170$ and $\sigma = 10$; a height of 185 becomes $\frac{185 - 170}{10} = 1.5$.
- **Min-max:**
  $$
  x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
  $$
  Every feature is squeezed into 0 to 1. Example: $x_{\min} = 150$, $x_{\max} = 200$; 185 becomes $\frac{185 - 150}{50} = 0.7$.

Both put all features on the same ruler. Remember: compute mean/std or min/max on the training set only (Q5).

**Why C and D are marked wrong.** A log transform changes the *shape* of a feature (it tames skewed data) but doesn't put features on a common scale. Rank normalisation replaces values with their rank order and throws away the actual distances between values. Neither is the standard answer for "make distances fair across features".

---

## Question 12 – Overfitting in high dimensions

**Correct:** A. Too few training samples relative to feature count. B. Excessive model complexity.
**Distractors:** C. Overly aggressive normalisation. D. Lack of noise in the training data.

**Why.** Overfitting = the model memorises the training data (including its quirks) instead of learning the general rule, so it does badly on new data.
- A: this is the **curse of dimensionality** (Lecture 1, section 5). With many features and few samples, there are countless ways to draw a boundary that fits the training points perfectly by accident. Remember 44 samples suffice in 4-D but $3.8 \times 10^{57}$ are needed in 128-D.
- B: a very flexible model (deep tree, high-degree polynomial kernel, k = 1 in k-NN) can bend around every training point.
- C: scaling features doesn't create overfitting. D: *more* noise makes overfitting worse, not less; lack of noise is not a cause.

---

## Question 13 – k-NN properties

**Correct (all four, after regrade):**
- A. Low training cost but high inference cost.
- B. Sensitive to irrelevant features.
- C. Boundaries are not always linear.
- D. Can approximate any decision boundary with enough data.

**Why.** Lecture 2, section 4.
- A: "training" is just storing the data (free). Predicting means measuring the distance to *every* stored sample, which is slow (high inference cost). The opposite of an SVM, which trains slowly and predicts fast.
- B: every feature goes into the distance with equal weight. A useless feature adds random noise to every distance and can flip the nearest neighbour.
- C: the boundary follows wherever the stored points sit, so it is usually jagged and curved, not a straight line.
- D: with enough data the stored points fill the space densely, so the neighbour-vote boundary can trace any shape.

---

## Question 14 – Feature scaling without normalisation

**Correct:** A. Large-range features dominate distances. B. k-NN predictions may be biased. C. SVM margins may distort.
**Distractor:** D. L2 regularisation automatically compensates.

**Why.** Lecture 2, section 6: "distances are sensitive to scale". If feature 1 runs 0–10,000 and feature 2 runs 0–1, then feature 1 decides almost every distance (A). k-NN uses those distances directly, so its neighbours and votes are driven by the big feature (B). The SVM margin is a distance, so it gets distorted the same way (C).

**Why D is wrong.** L2 regularisation (the penalty in ridge regression, Lecture 2's method map) shrinks weights to control complexity. It doesn't know or care about the units of your features. Scaling is a separate step you must do yourself.

---

## What to revise before Quiz 2 and the lab talk

1. **The scaling rule, in one breath:** fit the scaler on the training set, apply the same numbers to validation and test, never compute anything from the test set. Normalising test data on its own gives inconsistent scaling and untrustworthy (possibly inflated) results, but it is not "leakage into training". (Q5)
2. **Kernel trick = lift up, never squash down.** It lets the SVM act in a higher-dimensional space through dot products, giving curved boundaries without building new features. Dimensionality *reduction* is PCA/ICA/t-SNE. (Q7)
3. **Distance arithmetic:** $L_1 = \sum_i |\Delta_i|$, $L_2 = \sqrt{\sum_i \Delta_i^2}$, $\cos = \dfrac{x \cdot y}{\lVert x \rVert \lVert y \rVert}$. Watch for the trap where an option gives the *length of one vector* instead of the distance. (Q4)
4. **Why scaling matters for k-NN and SVM:** both use distances; big-range features dominate; z-score and min-max are the standard fixes. (Q8, Q11, Q14)
5. **Cross-validation:** it measures and selects; it doesn't prevent overfitting; more folds = more training data per fold, more cost, steadier estimate. (Q6, Q9)
6. **k-NN character:** free to train, slow to predict, hurt by useless features, curved boundaries. (Q3, Q13)
7. **Overfitting:** too many features for too few samples, or a model that's too flexible. (Q12)

## Key terms from this quiz

- **Kernel (linear, polynomial, RBF)** – the similarity function inside an SVM; ruler, bendy ruler, freehand.
- **One-vs-one / one-vs-rest** – $\frac{n(n-1)}{2}$ pairwise machines / $n$ one-per-class machines.
- **Score / confidence** – the signed distance a classifier reports; used to vote or pick the winner.
- **Outlier** – a sample far from the rest; L2 is hurt by it more than L1.
- **Mahalanobis distance** – Euclidean distance after correcting for each feature's spread and the correlations between features.
- **Normalisation / scaling** – putting features on a common ruler. **Z-score:** $x' = \frac{x - \mu}{\sigma}$. **Min-max:** $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$.
- **Data leakage** – test information influencing training.
- **Generalisation performance** – how well the model does on unseen data.
- **Variance of an estimate** – how much the measured score would jump around if you repeated the experiment.
- **Overfitting** – memorising training quirks instead of learning the rule.
- **Inference cost** – the time it takes to make a prediction.
- **L2 regularisation** – a penalty on big weights (ridge); controls complexity, does not fix scaling.
