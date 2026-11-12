# Lecture 2 Notes – Fundamentals of Machine Learning (the whiteboard lecture)

**Course:** D7041E Applied Artificial Intelligence
**Source:** `L2-fundamentals.pdf`, 16 handwritten whiteboard pages. Everything on these pages is what the teacher chose to write by hand, so treat all of it as important. Hard-to-read bits are marked [unclear: my best guess].

**What this lecture covers:**

1. The agent picture again, and how sensor data becomes a feature vector.
2. A map of machine learning methods.
3. Supervised learning: datasets and labels.
4. Memory-based learning (the idea behind k-nearest neighbours).
5. Distances: Manhattan ($L_1$), Euclidean ($L_2$), Minkowski.
6. Scale problems, and cosine similarity.
7. Measuring how good a classifier is: TP/TN/FP/FN, the confusion matrix, accuracy, precision, recall, $F_1$.
8. The multi-class confusion matrix.
9. Parameters vs hyperparameters, with the gradient-descent rule.
10. Doing experiments properly: UCI datasets, data splits, cross-validation, preprocessing without leakage, random seeds, automation.

---

## How the pieces fit

First the teacher says what the computer actually sees: a list of numbers per object (page 1). Then which methods exist (page 2). Then the simplest method: keep every example and compare new ones with the stored ones (pages 3–4). Comparing needs a distance (pages 5–6). Then: how do you know your classifier is any good (pages 7–9)? What knobs can you turn (pages 10–11)? And finally how to run experiments so you don't fool yourself (pages 12–16).

---

## 1. The agent and the feature vector (page 1)

The page is titled **"AI: Artificial intelligence"** and redraws the agent picture from Lecture 1, with more detail:

- A big **Agent** box on the left. Inside it: **"State & knowledge [unclear: of the] environment"**. The agent keeps its own internal picture of the world.
- An **Environment** blob on the right (with a smiley face in it).
- Several **sensors** (sensor 1 … sensor n) carry information from the environment into the agent. An arrow pointing at this part says **"Focus of this course"**. So: this course is about what happens between the sensors and the decision.
- **Actuators** carry actions back out to the environment.
- An arrow from the state box leads down to the word **"grounding"**. Grounding means tying the agent's internal symbols to actual sensor readings, so "cat" inside the agent is connected to real pixels.

Underneath, the key formula:

$$
\mathbf{X} = \{\, x_1,\ x_2,\ \ldots,\ x_n \,\}
$$

with arrows to the words **"features"** and **"feature vector"**.

**In plain words.** A **feature** is one measured property, written as a number: weight, height, the brightness of one pixel. A **feature vector** is all the features of one object lined up. $n$ is how many features there are. Think of a patient's chart: temperature, blood pressure and pulse are three features; the chart is the feature vector.

---

## 2. A map of machine learning methods (page 2)

The teacher drew **ML** in the middle with two branches.

### Branch 1: Classical ML — for "well-defined features"

Use when your features already mean something (age, price, temperature).

- **Supervised** (answers given):
  - **K-NN** – k-nearest neighbours (this lecture).
  - **Decision trees** – a learned flow chart of yes/no questions.
  - **Regression** – predicting a number rather than a class. Listed: **linear** (fit a straight line) and **Ridge** (linear, but with a penalty that keeps the coefficients small).
- **Unsupervised** (no answers):
  - **DB-Scan** [unclear: DBSCAN] – clustering by finding dense areas of points.
  - **K-means** – clustering around K centre points.
  - **t-SNE** [unclear: written "t-sne"], **PCA**, **ICA** – bracketed together and labelled **"dimensionality reduction methods"**. They squash many features into a few (for plotting, or to fight the curse of dimensionality).

### Branch 2: ANN (artificial neural networks) — for "no obvious features", "complex data"

Use when the raw data (pixels, sound, text) has no hand-made features.

- **Supervised:** **Conv. Neur. Nets (CNN)** for images; **Graph NN**; **Transformers** (the engine behind modern language models); **MLP** (multi-layer perceptron, the plain layered network).
- **Unsupervised:** **Autoencoders** (learn to compress and rebuild their input); **Self-Organised Maps** (later in the course).
- **Recurrent NN** (networks with memory, for sequences): **LSTM**; **Echo-State Net** (Lab 4).

---

## 3. Supervised learning: the dataset (page 3)

The teacher writes: **reference base ≡ Dataset.** The dataset is your collection of examples. In supervised learning each example also carries **Category ≡ label**, the right answer.

The handwritten example:

| $x_1$ | $x_2$ | Category (label) |
|---|---|---|
| 4 | 2.5 | cat |
| 2 | 3.5 | dog |
| 1 | 1 | [unclear: Plane] |

Each row is one **sample**: two features plus one label. We'll reuse this tiny table below.

---

## 4. Memory-based learning (page 4)

**The idea.** Don't build anything. Just store all the examples. When a new one comes, compare it with the stored ones and copy the label of the most similar.

**The picture.** Axes $f_1$ and $f_2$ ("$f_1, f_2 \rightarrow$ features"). Plus signs (+) are **Category 2**, circles (o) are **Category 1**. Each symbol is a **data sample**. A **new sample** appears among them. Which category? Look at who is closest.

Everyday version: you move into a street and want to guess how people there vote. You ask your nearest neighbours and go with the majority.

**1-NN in steps:**

1. Store every training feature vector with its label.
2. Take the new sample.
3. Measure the distance from it to *every* stored sample.
4. Find the smallest distance.
5. Copy that sample's label.

**k-NN:** same, but in step 4 keep the $k$ closest, and in step 5 let them vote. The number $k$ is a setting you choose (see section 9).

---

## 5. Distance (page 5)

To compare samples you need a number for "how far apart". The sketch: two points $x$ and $y$ in the $f_1$–$f_2$ plane. Their coordinates are written $f_1^{x}, f_2^{x}$ (for $x$) and $f_1^{y}, f_2^{y}$ (for $y$). So $f_i^{x}$ means "feature number $i$ of sample $x$".

### $L_1$ – Manhattan distance

$$
L_1(x, y) = \sum_{\text{all features } i} \left| f_i^{x} - f_i^{y} \right|
$$

Steps: for each feature, subtract; drop the minus sign; add everything up.

Why "Manhattan"? In a grid city you can't walk through buildings. You go along one street, then the other. The dashed lines in the sketch are those two legs.

### $L_2$ – Euclidean distance

$$
L_2(x, y) = \sqrt{\sum_{\text{all features } i} \left( f_i^{x} - f_i^{y} \right)^2 }
$$

Steps: subtract; square each difference (which also kills the minus sign); add up; take the square root. This is the straight diagonal line in the sketch, Pythagoras in disguise.

**Why the square root?**

*Aim:* see why $L_2$ ends with a square root instead of stopping at the sum of squares.
*Where we're heading:* a case where we know the true straight-line distance by eye, so we can see which version of the formula matches it.

Take two points whose differences are 3 and 4. Squares: 9 and 16. Sum: 25. If you stopped here you'd say "distance 25", but draw it: the two legs 3 and 4 form a right-angled triangle whose diagonal is 5, not 25. The square root undoes the squaring and brings the answer back to the same units as the features (metres, not square metres). Squaring was only there to remove the minus signs and to let Pythagoras work; the root cleans up afterwards.

*Reached:* with the root, the formula gives 5, the length you can measure with a ruler. Without it, 25, which is in the wrong units. ✓

### Minkowski distance – the general recipe

$$
L_p(x, y) = \left( \sum_{i} \left| f_i^{x} - f_i^{y} \right|^{p} \right)^{1/p}
$$

Steps: absolute differences; raise each to the power $p$; add; take the $p$-th root.

- $p = 1$ → $L_1$ (power 1 and first root change nothing).
- $p = 2$ → $L_2$ (squares and a square root).

So $L_1$ and $L_2$ are two members of one family, and the lab's "L1 norm / L2 norm" means exactly these.

### Worked example

New sample $q = (3.5,\ 3)$. Dataset: cat $(4,\ 2.5)$, dog $(2,\ 3.5)$, plane $(1,\ 1)$.

**$q$ to cat:**

1. Differences: $3.5 - 4 = -0.5$ and $3 - 2.5 = 0.5$.
2. Drop signs: $0.5$ and $0.5$.
3. $L_1 = 0.5 + 0.5 = \mathbf{1.0}$.
4. Squares: $0.25 + 0.25 = 0.5$. $L_2 = \sqrt{0.5} \approx \mathbf{0.707}$.

**$q$ to dog:**

1. Differences: $1.5$ and $-0.5$. Drop signs: $1.5$ and $0.5$.
2. $L_1 = \mathbf{2.0}$.
3. Squares: $2.25 + 0.25 = 2.5$. $L_2 = \sqrt{2.5} \approx \mathbf{1.581}$.

**$q$ to plane:**

1. Differences: $2.5$ and $2$.
2. $L_1 = \mathbf{4.5}$.
3. Squares: $6.25 + 4 = 10.25$. $L_2 = \sqrt{10.25} \approx \mathbf{3.202}$.

Both distances say the nearest is the cat, so 1-NN predicts **cat**. They don't always agree, which is why the lab asks you to try both.

---

## 6. Scale, and cosine similarity (page 6)

Left side: **"Distances are sensitive to scale."** The sketch shows three arrows $d_1, d_2, d_3$ from the origin. $d_2$ is long and $d_1$ is short, but they point almost the same way.

**Why this matters.** If one feature is in millimetres and another in kilometres, the millimetre one produces huge numbers and takes over the distance. Also, doubling all the features of a sample changes every distance even though the *pattern* didn't change.

Quick check: $L_2$ between cat $(4, 2.5)$ and dog $(2, 3.5)$ is $\sqrt{4 + 1} \approx 2.236$. Double the cat to $(8, 5)$: $L_2$ to dog becomes $\sqrt{36 + 2.25} \approx 6.185$. Nearly three times bigger, for no real reason.

Right side: **"Similarity focuses on shared structure & alignment."** The cosine formula:

$$
\cos(d_1, d_2) = \frac{ f_1^{d_1} f_1^{d_2} + f_2^{d_1} f_2^{d_2} }{ \lVert f^{d_1} \rVert \cdot \lVert f^{d_2} \rVert }
$$

Top: the **dot product** (multiply matching features, add up). Bottom: the two vectors' **lengths** multiplied. A vector's length (norm, written $\lVert \cdot \rVert$) is the square root of the sum of its squared features:

$$
\lVert v \rVert = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

The result is the cosine of the angle between the two arrows: $1$ = same direction, $0$ = right angle (nothing shared), $-1$ = opposite.

Everyday version: two shopping baskets. One has 2 apples and 1 loaf; the other 20 apples and 10 loaves. By distance they're far apart. By cosine they're identical, because the *proportions* match.

**Worked example:** cat $d_1 = (4, 2.5)$, dog $d_2 = (2, 3.5)$.

1. Dot product: $4 \times 2 + 2.5 \times 3.5 = 8 + 8.75 = 16.75$.
2. Length of $d_1$: $\sqrt{16 + 6.25} = \sqrt{22.25} \approx 4.717$.
3. Length of $d_2$: $\sqrt{4 + 12.25} = \sqrt{16.25} \approx 4.031$.
4. Multiply lengths: $4.717 \times 4.031 \approx 19.014$.
5. $\cos = 16.75 / 19.014 \approx \mathbf{0.881}$.

Now double the cat to $(8, 5)$: dot $= 33.5$, length $\approx 9.434$, $\cos = 33.5 / (9.434 \times 4.031) \approx \mathbf{0.881}$. Unchanged. That's the teacher's point: cosine ignores size and looks at direction.

**Why doubling cancels out.**

*Aim:* explain, not just observe, why cosine similarity did not move when the cat vector was doubled.
*Where we're heading:* showing that any stretch factor appears once on top of the fraction and once on the bottom, so it divides out.

Doubling the cat doubles every term in the dot product ($16.75 \rightarrow 33.5$) and doubles the cat's length ($4.717 \rightarrow 9.434$). The top doubled, the bottom doubled, and $2 / 2 = 1$. Any stretch you apply to a vector appears once upstairs and once downstairs, so it always cancels. That is the mechanical reason cosine "focuses on alignment".

---

## 7. Assessing a classifier (pages 7–8)

### Words first (page 7)

**Binary classification** = two classes. Call class 1 the **positive** class and class 2 the **negative** class. (Example: "spam" = positive, "not spam" = negative.)

Correctly classified:
- **True positive (TP)** – really positive, predicted positive.
- **True negative (TN)** – really negative, predicted negative.

Error cases:
- **False positive (FP)** – really negative, predicted positive. A false alarm.
- **False negative (FN)** – really positive, predicted negative. A miss.

### The table (page 8), which the teacher calls a "truth table" (most people say confusion matrix)

As drawn:

|  | P | N |
|---|---|---|
| **P** | TP | FN |
| **N** | FP | TN |

> **⚠ Something to check with your teacher.** The arrows on the page say the **row** labels are the "outcome of classifier" and the **column** labels are the "desired outcome" (truth). But if that were so, the top-right cell (classifier says P, truth is N) would be a false *positive*, FP, not FN. The cells as written match the opposite reading: **rows = true class, columns = predicted class**. Either the arrows or the two cells are swapped. The standard layout that matches the cells is:
>
> |  | predicted P | predicted N |
> |---|---|---|
> | **truly P** | TP | FN |
> | **truly N** | FP | TN |
>
> scikit-learn (used in the lab) uses rows = true, columns = predicted.

### The four scores (page 8, right side)

**Accuracy** – share of everything that was right:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

[unclear: the handwritten denominator reads "TP + FN + FP + TN"; same sum.]

**Precision** – of the things you called positive, how many really were:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall** – of the things that really are positive, how many you found:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**$F_1$ score** – one number that is high only when precision *and* recall are both high:

$$
F_1 = \frac{2 \cdot \text{Recall} \cdot \text{Precision}}{\text{Recall} + \text{Precision}}
$$

Why bother with more than accuracy? If 99% of emails are fine, a filter that says "fine" to everything has 99% accuracy and catches no spam. Precision and recall reveal that; $F_1$ summarises both.

**Why $F_1$ is that odd fraction and not a plain average.**

*Aim:* understand what the $F_1$ formula is designed to do.
*Where we're heading:* an example where precision is perfect and recall is terrible, so we can see that a plain average would reward it and $F_1$ would not. Then we read the design off the formula.

Imagine a filter that flags only one email all year, and that one really is spam. Precision $= 1/1 = 1.0$ (perfect). But it let 49 other spam emails through, so recall $= 1/50 = 0.02$. A plain average would give $(1.0 + 0.02)/2 = 0.51$, which sounds half-decent for a filter that catches nothing. $F_1$ gives

$$
F_1 = \frac{2 \times 1.0 \times 0.02}{1.0 + 0.02} = \frac{0.04}{1.02} \approx 0.039 .
$$

That is the honest number. The formula is built so that the *smaller* of the two dominates: multiply on top (one tiny factor makes the product tiny), and the 2 is there so that when precision and recall are equal, $F_1$ equals them ($2 \times 0.8 \times 0.8 / 1.6 = 0.8$). Mathematicians call this the harmonic mean; you can just remember "$F_1$ punishes you for neglecting either one".

*Reached:* plain average $0.51$ versus $F_1 \approx 0.04$ for a useless filter. The formula does what we wanted: it stays low unless both precision and recall are high. ✓

### Worked example

100 emails. $TP = 40$, $FN = 10$, $FP = 5$, $TN = 45$.

1. $\text{Accuracy} = \dfrac{40 + 45}{100} = \mathbf{0.85}$.
2. $\text{Precision} = \dfrac{40}{40 + 5} = \dfrac{40}{45} \approx \mathbf{0.889}$. Of 45 flagged, 40 were really spam.
3. $\text{Recall} = \dfrac{40}{40 + 10} = \dfrac{40}{50} = \mathbf{0.80}$. Of 50 real spam, 40 caught.
4. $F_1 = \dfrac{2 \times 0.80 \times 0.889}{0.80 + 0.889} = \dfrac{1.422}{1.689} \approx \mathbf{0.842}$.

---

## 8. The multi-class case (page 9)

With classes $cl_1 \ldots cl_n$ the table becomes $n \times n$. Rows are the classes, columns are the classes. The diagonal cells are the correct predictions. The teacher boxes the cell at row $i$, column $i$ and writes:

- **FN** ⇒ sum along **row $i$** without the $i$-th element. (Things that *are* class $i$ but were called something else.)
- **FP** ⇒ sum along **column $i$** without the $i$-th element. (Things called class $i$ that are really something else.)
- TP for class $i$ = the diagonal cell itself.

If $C_{rc}$ is the count in row $r$, column $c$:

$$
TP_i = C_{ii}, \qquad
FN_i = \sum_{c \neq i} C_{ic}, \qquad
FP_i = \sum_{r \neq i} C_{ri}
$$

This matches rows = true, columns = predicted.

**Worked example** (30 test samples, rows = true, columns = predicted):

|  | pred cat | pred dog | pred plane |
|---|---|---|---|
| **true cat** | 8 | 1 | 1 |
| **true dog** | 2 | 7 | 1 |
| **true plane** | 0 | 1 | 9 |

For **dog** ($i = 2$):

1. $TP = 7$.
2. $FN$ = row without diagonal $= 2 + 1 = 3$.
3. $FP$ = column without diagonal $= 1 + 1 = 2$.
4. $\text{Precision(dog)} = 7/9 \approx 0.778$. $\text{Recall(dog)} = 7/10 = 0.70$.

Overall accuracy = diagonal sum ÷ total $= (8 + 7 + 9)/30 = \mathbf{0.80}$.

---

## 9. Parameters vs hyperparameters (pages 10–11)

### Two different things called "parameters" (page 10)

**Model parameters** – **learned** from the data.
- ANN: the **weights**.
- Regression: the **regression coefficients**.

**Hyperparameters** – **configuration choices** you make before training. They **control how the algorithm learns, how complex the model is, how the model processes data**.

Baking analogy: oven temperature and baking time are settings you choose (hyperparameters). How the dough actually rises is the learned result (parameters).

### Examples (page 11)

**k-nearest neighbour:** the number $k$, circled, with an arrow to "hyperparam". The sketch shows a new sample, a cloud drawn around it holding some + and o points, and the question "class?". How big you draw the cloud is $k$.

**MLP:** a sketch of an input layer, hidden layers of little circles (**neurons**), and an **output layer**. Arrows point to **"# layers"** and **"# neurons"** as hyperparameters.

And at the bottom, the learning rule:

$$
w_{t+1} = w_t - \eta \, \nabla L(w_t)
$$

[unclear: the circled symbol is read as $\eta$ (eta), labelled "learning rate".]

What each symbol means:
- $w_t$: the weights now (at step $t$).
- $L(w)$: the **loss**, a number for "how wrong the model is". Lower is better.
- $\nabla L$: the **gradient**, the direction in which the loss climbs fastest. The minus sign means "go the other way, downhill".
- $\eta$: the **learning rate**, how big a step to take. A hyperparameter.

Foggy-hill picture: you feel which way the ground slopes (gradient) and step downhill (minus). Step size is $\eta$. Too small and you crawl; too big and you leap over the valley and bounce around.

**Worked example** (added so the formula feels real). Loss $L(w) = (w - 3)^2$, lowest at $w = 3$. Gradient $\nabla L(w) = 2(w - 3)$. $\eta = 0.25$. Start $w_0 = 0$.

1. Gradient $= 2(0 - 3) = -6$. $w_1 = 0 - 0.25 \times (-6) = \mathbf{1.5}$.
2. Gradient $= 2(1.5 - 3) = -3$. $w_2 = 1.5 + 0.75 = \mathbf{2.25}$.
3. Gradient $= -1.5$. $w_3 = \mathbf{2.625}$.
4. Gradient $= -0.75$. $w_4 = \mathbf{2.8125}$.

It creeps toward 3 and the steps shrink as the ground flattens. With $\eta = 1.5$ the first step would land at $w = 9$ and things would explode. That's why $\eta$ matters.

**Why the minus sign, using the same numbers.**

*Aim:* convince ourselves that subtracting the gradient always moves downhill, whichever side of the valley we start on.
*Where we're heading:* two starts, one left of the minimum and one right of it, both ending up closer to the minimum after one step.

At $w = 0$ the gradient is $-6$. A negative gradient means "the loss goes *down* as $w$ goes *up*". So we need $w$ to increase. Subtracting a negative number ($0 - 0.25 \times (-6) = +1.5$) does exactly that. Try starting on the other side, at $w = 5$: gradient $= 2(5 - 3) = +6$, positive, meaning the loss rises as $w$ rises, so we need $w$ to fall. New $w = 5 - 0.25 \times 6 = 3.5$. Down it goes. The minus sign automatically picks the downhill direction whichever side of the valley you are on. With a plus sign you would climb instead.

*Reached:* from $w = 0$ we moved to $1.5$ (closer to 3); from $w = 5$ we moved to $3.5$ (closer to 3). Downhill both times. ✓

**Where the gradient came from.** $L(w) = (w - 3)^2$ is "the square of how far $w$ is from 3". Its derivative, by the chain rule, is

$$
\frac{dL}{dw} = 2\,(w - 3) \times 1 = 2(w - 3).
$$

In a real network the loss depends on thousands of weights and the gradient is the list of all those derivatives at once, but each entry means the same thing: "if I nudge this weight up, does the loss go up or down, and how fast?"

---

## 10. Experimenting with data (pages 12–16)

### UCI datasets (page 12)

- **UCI dataset: 121 datasets** (the collection behind Lab 1 Part 4).
- **19 "separated" datasets** – come as **2 files: train and test files**.
- **102 "not separated"** – come as **1 file**; you must split them yourself. ($19 + 102 = 121$.)

### Data splits (page 13)

The sketch has a vertical line splitting the data into two big parts.

**Left: Train.** Split again into
- **train 50%** – "train model for different configurations of hyperparams";
- **validation part 50%** – "test for each config".

**Right: Test splits.** Note beside it: **"Train of full train set"**: after you've picked the best configuration, retrain on the *whole* train part, then score on the test part.

So three jobs:
1. **Training set** – fit the model.
2. **Validation set** – compare hyperparameter settings, pick the best.
3. **Test set** – touch it **once**, at the very end, for the final number.

Why so strict? If you tune on the test set, you've tuned to that specific set and your score is flattering. The test set is a sealed exam paper. (The 50/50 split is the teacher's example; the lab uses 80:20 elsewhere. The proportions are a choice.)

### Cross-validation when data is small (page 14)

> **"if data is small, use cross-fold validation technique"** (usually called k-fold cross-validation)

The sketch: a tall column from "Sample 0" to "Sample 10000", cut into **fold 1, fold 2, …, fold 5**.

1. Split the *training* data into $n$ equal folds.
2. **Use $n - 1$ folds for training and 1 for validation.**
3. Rotate the validation fold **until all folds are used for validation**.
4. **Average results.** That's the score for this hyperparameter setting:
   $$
   \text{CV score} = \frac{1}{n} \sum_{j=1}^{n} \text{score}_j
   $$
5. Repeat for every setting you want to compare; pick the best.
6. Retrain on all training data with that setting; test once on the test set.

Why? A single small validation set gives a noisy score. Cross-validation lets every sample be validation once, so you use all your data and get a steadier estimate. Typical: 3, 5 or 10 folds.

**Lab example (3-fold, choosing $k$ for k-NN):** folds A, B, C; candidates $k = 1, 3, 5$.
1. For $k = 1$: train on B+C, validate on A; train on A+C, validate on B; train on A+B, validate on C. Average the three scores.
2. Same for $k = 3$ and $k = 5$.
3. Pick the $k$ with the best average.
4. Run once on the test set with that $k$. Report that accuracy.

### Preprocessing and data leakage (page 15)

> **"Do configuration of data preprocessing separately for train & test sets, to avoid data leakage."** (underlined)

**Preprocessing** = cleaning up features before training, mostly scaling them (remember: distances are sensitive to scale). **Data leakage** = information from the test set sneaking into training. Example: if you compute the mean and spread for scaling using train *and* test together, training has secretly seen the test data. Do it right: work out the scaling numbers from the training set only, then apply those same numbers to the test set. For z-score scaling that means

$$
x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

applied to training, validation **and** test data, with $\mu$ and $\sigma$ computed from the training set alone.

### Random seeds (page 15)

> **"In algorithms with random initialization (ANN), repeat all experiments with different SEED values of RNG."**

RNG = random number generator. The **seed** is the starting value that fixes which "random" numbers come out; same seed, same numbers. Neural networks start from random weights, so one run can be lucky. Run several times with different seeds and **average**. Record the seeds so anyone can repeat the run.

### Important (page 16)

> - **Automate your experiments.**
> - **Save results + configuration for repeatability.**

Write a script that loops over datasets, models, settings and seeds, and saves every result next to the exact settings that produced it. Future-you will be grateful, and the lab report requires it.

---

## Key points to remember

- The computer sees a feature vector $\mathbf{X} = \{x_1, \ldots, x_n\}$ per object. $n$ = dimension.
- Classical ML for clean features; neural networks for raw, complex data. Both have supervised and unsupervised members.
- Memory-based learning: store everything, compare, copy the nearest label. k-NN lets $k$ neighbours vote.
- $L_1 = \sum |{\Delta}|$; $L_2 = \sqrt{\sum \Delta^2}$; both are Minkowski with $p = 1, 2$.
- Distances depend on scale, so scale your features. Cosine similarity ignores scale and measures direction.
- TP/TN right, FP false alarm, FN miss. $\text{Precision} = \frac{TP}{TP+FP}$, $\text{Recall} = \frac{TP}{TP+FN}$, $F_1 = \frac{2PR}{P+R}$.
- Multi-class: FN = row minus diagonal, FP = column minus diagonal.
- Parameters are learned (weights); hyperparameters are chosen ($k$, layers, neurons, learning rate).
- Gradient descent: $w_{t+1} = w_t - \eta \nabla L(w_t)$.
- Train / validation / test. Never tune on test. Small data → k-fold cross-validation.
- Fit preprocessing on train only (no leakage). Repeat random algorithms with several seeds. Automate and save everything.

## Key terms

- **Feature / feature vector / dimension** – one measurement / all of them / how many.
- **Grounding** – tying internal symbols to sensor data.
- **Dataset (reference base) / label (category)** – all examples / the right answer per example.
- **Memory-based learning** – store and compare.
- **k-NN / 1-NN** – vote of $k$ nearest neighbours / just the nearest.
- **$L_1$ (Manhattan), $L_2$ (Euclidean), Minkowski** – grid distance, straight-line distance, the general family.
- **Norm $\lVert v \rVert$** – a vector's length.
- **Dot product / cosine similarity** – multiply-and-add / dot product over lengths, the angle.
- **TP, TN, FP, FN** – true positive, true negative, false positive, false negative.
- **Confusion matrix ("truth table")** – true vs predicted table.
- **Accuracy, precision, recall, $F_1$** – the four scores above.
- **Model parameters / hyperparameters** – learned values / chosen settings.
- **Loss, gradient, learning rate** – how wrong; steepest-climb direction; step size.
- **Train / validation / test set** – fit / choose settings / final score once.
- **k-fold cross-validation** – rotate the validation fold, average.
- **Preprocessing / data leakage** – feature clean-up / test info sneaking into training.
- **Seed / RNG** – start value of the random generator / the generator.

## Practice questions

**1. $L_1$ and $L_2$ between $(1, 4, 2)$ and $(3, 1, 2)$?**
Differences $2, 3, 0$. $L_1 = 5$. $L_2 = \sqrt{4 + 9 + 0} = \sqrt{13} \approx 3.606$.

**2. Classify $(1.5, 1.5)$ with 1-NN and $L_1$ using the cat/dog/plane table.**
To cat $2.5 + 1 = 3.5$; dog $0.5 + 2 = 2.5$; plane $0.5 + 0.5 = 1.0$. Nearest is plane.

**3. $TP = 30$, $FP = 20$, $FN = 10$, $TN = 40$. Accuracy, precision, recall, $F_1$?**
Accuracy $0.70$; precision $30/50 = 0.60$; recall $30/40 = 0.75$; $F_1 = \frac{2 \times 0.6 \times 0.75}{1.35} \approx 0.667$.

**4. Cosine between $(3, 4)$ and $(6, 8)$?**
Dot $50$; lengths $5$ and $10$; $\cos = 50/50 = 1$. Same direction, scale ignored.

**5. Why fit the scaler on the training set only?**
Otherwise test-set statistics leak into training and your test score is no longer honest.

**6. One gradient step: $L(w) = (w - 5)^2$, $w = 2$, $\eta = 0.1$.**
Gradient $= 2(2 - 5) = -6$. New $w = 2 - 0.1 \times (-6) = 2.6$.

**7. In 5-fold cross-validation, how many models do you train per hyperparameter setting, and on what share of the training data each?**
Five models, each on $4/5 = 80\%$ of the training data.
