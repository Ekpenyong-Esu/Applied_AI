# Lecture 1 Notes – Course Introduction, What AI Is, and Why Dimensions Matter

**Course:** D7041E Applied Artificial Intelligence, Luleå University of Technology (lecturer Evgeny Osipov)
**Source:** `L1-D7041E(Intro)_LP2_2026.pdf`, 45 slides

**What this lecture covers:**

1. How the course works (labs, quizzes, grades).
2. What "intelligence" is, and what we are trying to copy into a machine.
3. The idea of an agent.
4. What learning really means, and the main types of learning.
5. The curse of dimensionality (why lots of features is a problem).
6. The blessing of dimensionality (why lots of dimensions can also help). This is the lecturer's own research area.

---

## How this lecture connects to the rest

This lecture is the "why" behind everything else. Lecture 2 teaches the basic machinery (data, k-NN, distances, metrics, honest testing). Lecture 3 teaches one proper learning model (the SVM). Lab 1 makes you do it all. This lecture tells you what the machinery is *for* and warns you about the one big trap (too many dimensions, too little data).

---

## 1. How the course works

**What you should be able to do at the end.** In plain words: know the foundations of AI systems, know some AI methods deeply, build learning systems that serve people and sustainable goals, design and implement learning components, compare AI models critically, and make predictions even when information is limited.

**Four blocks.** Each block has a theory week (lectures plus a quiz) and a practice week (a lab).

| Block | Theory | Lab |
|---|---|---|
| 1 | Introduction, basics of machine learning, word embeddings. Quiz 1 | Lab 1: ML fundamentals and systematic experiments |
| 2 | Natural language processing, hyperdimensional representations. Quiz 2 | Lab 2: NLP and text embeddings |
| 3 | Self-organising maps, multi-layer perceptron with back-propagation. Quiz 3 | Lab 3: SOM and MLP |
| 4 | Time series, recurrent neural networks. Quiz 4 | Lab 4: Echo State Networks |

**Maths you are assumed to know:** probability, linear algebra, calculus, optimisation, statistics, computational geometry. Don't panic; the lectures re-explain what they need.

**Book:** Russell and Norvig, *Artificial Intelligence: A Modern Approach*. Most practical information is in the "README! Essential information" page on Canvas, which gets updated during the course.

**Labs.**
- Four labs, groups of 2 or 3.
- Deadline: end of Friday of the practice week.
- You pass a lab by **talking it through with a teaching assistant**. Handing in a report is not enough.
- The talk is individual. The TA checks whether *you* understand, not whether an AI tool wrote it. "My partner did that part" does not work.

**Quizzes.**
- Four of them, online in Canvas, graded automatically.
- Open from Wednesday midnight to Friday midnight of the theory week. Once you start, 1.5 hours.
- Multiple choice and multiple answer.
- **All four must be done to pass the quiz part.**

**Grading.** Two parts: Labs (4 hp) and Quizzes (3.5 hp), each graded 3, 4 or 5.
- 5 in both → course grade 5.
- One 3 and the other 4 or 5 → course grade 4.
- 3 in both → course grade 3.

---

## 2. What is AI, really?

**A bit of history.** The modern story starts in the 1950s, when mathematicians and computer scientists began to say "maybe thinking can run on a computer". The name "artificial intelligence" was invented in 1956 at a meeting in Dartmouth College (McCarthy, Minsky, Newell, Samuel, Simon).

**What people say when asked "what is AI?":** things humans normally do but a machine does; solutions made by a computer; LLMs (large language models like chat assistants); deep learning (the lecturer jokes: is there "shallow" learning?); machine learning.

**The real question:** what exactly are we trying to re-create? The lecturer answers with six things humans do well.

1. **We are goal-directed.** We don't do random things. Our thoughts point at what we want. We don't try every option; we try the sensible ones.
2. **We adapt to the situation.** Plan A fails, we switch to Plan B. Our thinking is shaped by the world around us, not only by a plan made in advance.
3. **We carry a huge amount of knowledge.** Try writing down *everything* you know about tying shoelaces. You will give up before you finish. Skills hide enormous detail.
4. **We cope with a messy, detailed world using very little information.** Give someone a rough sketch map and vague directions and they still find the treasure.
5. **We form symbols and abstractions.** See a handful of chickens (a photo, a cartoon, a roast) and you recognise the next chicken you meet, even though it looks different. We build a general idea from a few examples. (The slide says "spices"; it means "species".)
6. **We learn from experience.** We are not born knowing everything. We learn all through life. **This is the one the course is about.**

---

## 3. The agent

> An **agent** is anything that **perceives** its **environment** through **sensors** and **acts** on it through **actuators**.

Picture a robot vacuum. Sensors: bump sensors, a camera. Actuators: wheels and a brush. In the middle sits a box with a question mark: the part that decides. The course is about that box, and especially about how it learns from what the sensors deliver.

---

## 4. What does it mean to learn?

**Start with a picture.** A program that is bad at a task at first but gets better each time it practises. That's learning.

**Learning means getting new information.** Nobody can have *all* the information (that would be knowing the whole universe). In made-up worlds like chess, complete information is possible.

**Is memorising the same as learning?** No, not always. Think of your own studies: cramming facts is not the same as understanding. The slide puts it this way:

> Learning is about processing information and extracting general rules (the general structure of a phenomenon) from particular examples.

**But** remembering past cases and finding similar ones ("retrieving analogues from memory") is still important. That is exactly what the nearest-neighbour method in Lecture 2 does.

**Why not just write down all the rules?** You can. That is what **expert systems** are: big lists of "if this, then that" rules. Two problems:

- Writing a complete rule list is extremely hard. Some rules can't be put into words; they live as a feeling built from lots of experience.
- Even a complete description isn't enough. If someone handed you a detailed written guide to karate moves, you still could not do karate.

**The semantic gap.** The hard part is connecting what the sensors register (pixels, sounds) to the abstract rules ("this is a cat"). That missing connection is called the *semantic gap*. Rules without that link are useless. Learning a formula and matching its parts to a problem (like the quadratic formula) solves only that exact kind of problem. Is that learning or just programming? The lecturer notes that for humans this is still the most common kind of learning.

### 4.1 The AI definition of learning

> **Learning by examples:** given a collection of examples (a *training set*), be able to solve new problems of the same kind.

### 4.2 Three main types of learning

**Supervised learning.** The correct answers are given. It's like tuning a program until it gives the best result. Formally:

> Given a training set of $N$ input–output pairs
> $$
> (x_1, y_1),\ (x_2, y_2),\ \ldots,\ (x_N, y_N),
> $$
> where each $y$ was produced by an unknown function $y = f(x)$, find a function $h$ that approximates $f$.

The function $h$ is called a **hypothesis**. It **generalises well** if it predicts $y$ correctly for *new* examples it has never seen. That last part is the whole point: anyone can memorise the training set.

**Unsupervised learning.** Only the problem is given, no correct answers. Useful for autonomous systems whose answers don't need to be explained in the programmer's words. The most common task is **clustering**: finding natural groups in the data. The slide shows a scatter of points that clearly form three blobs, with no labels telling you which blob is which.

**Reinforcement learning.** No answers, only a signal saying "good" or "bad" (a reward or punishment). **The environment is the trainer.** The slide's example: honeybees choose between two visual patterns. The right pattern gives sugar water, the wrong one gives bitter quinine. After 30 practice trials, the bees picked correctly about 60–70% of the time on new tests, above the 50% you'd get by guessing, in three of the four tests.

### 4.3 Other ways to slice learning

- **Passive vs active.** In active learning the learner itself goes looking and **picks the examples worth learning from**.
- **Non-incremental vs incremental.** Non-incremental: all examples arrive at once. Incremental: examples keep arriving and the learner keeps updating. Incremental learners are more flexible and autonomous but less predictable.
- **Meta-learning.** Learning how to learn: finding the right learning algorithm for a domain. The lecturer calls it the hardest kind.

### 4.4 Learning = building a model

Learning about something means building a **model**: a thing that takes the agent's inputs and produces an output (a category, the next value, and so on). In ML words: *fit a model to a dataset of observations*.

- More data → a more accurate model.
- Ideally the data should cover the whole range of inputs you care about. Then every prediction is an **interpolation** (a guess between known points, safe) rather than an **extrapolation** (a guess outside known territory, risky).

---

## 5. The curse of dimensionality

**Dimension** just means "number of features". A sample with 3 measurements lives in 3-D space; a $28 \times 28$ pixel image lives in 784-D space.

**The key comparison from the slides:**

> It is very different to analyse 10,000 observations of 3 variables each than 100 observations of 50 variables each.

- 10,000 points in 3-D form visible shapes: clouds you can measure (main directions, spread).
- 100 points in 50-D "do not represent anything specific" because 100 points are far too few to fill a 50-D space.

**The rule:** the number of samples you need grows **exponentially** with the number of dimensions. Roughly,

$$
\text{samples needed} \;\propto\; c^{\,d} \qquad (c > 1,\ d = \text{dimension}),
$$

so each extra dimension multiplies the requirement by the same factor.

**A striking example from the slide.** To learn a bell-shaped (Gaussian) distribution with a k-NN classifier:

| Dimensions | Observations needed |
|---|---|
| 4 | about $44$ |
| 128 | at least $3.8 \times 10^{57}$ |

That second number is bigger than the number of atoms in the Earth. So in high dimensions your data is *always* sparse.

**Everyday picture.** Cover a 1-metre line with dots every 10 cm: $10$ dots. Cover a 1 m × 1 m square the same way: $10 \times 10 = 100$ dots. A cube: $10^3 = 1{,}000$. Every extra dimension multiplies the need by 10. In general, $10^{d}$ dots for $d$ dimensions.

### 5.1 A feeling for NP-completeness

A side trip to show what "exponential" does to you. Compare two algorithms on an $N$-bit number.

- **Algorithm 1:** 5 million operations per bit, so $5 \times 10^{6} \cdot N$ operations total. Doubling $N$ doubles the time. (Linear.)
- **Algorithm 2:** tests whether the number is prime by trying divisors, about $2^{N/2}$ operations. Adding 2 bits doubles the time. (Exponential.)

Running times from the slide (1 GHz means $10^{9}$ operations per second):

| $N$ (bits) | 10 | 20 | 80 | 120 |
|---|---|---|---|---|
| Algorithm 1 at 1 GHz | < 0.1 s | 0.1 s | 0.4 s | 0.6 s |
| Algorithm 2 at 1 GHz | < 0.1 s | < 0.1 s | 1100 s | 37 years |
| Algorithm 2 at 1 THz (1000× faster) | < 0.1 s | < 0.1 s | 1.1 s | 13 days |

**Let's check two entries so the table isn't magic.**

*Aim:* reproduce the "0.6 s", "37 years" and "13 days" cells from the two formulas, so you could rebuild the table yourself in an exam.
*Where we're heading:* operations count ÷ operations per second = seconds, then convert seconds to years or days.

*Algorithm 1, $N = 120$:*

1. Operations $= 5 \times 10^{6} \times 120 = 6 \times 10^{8}$.
2. Time $= \dfrac{6 \times 10^{8}}{10^{9}} = 0.6$ seconds. ✓

*Algorithm 2, $N = 120$, at 1 GHz:*

1. Operations $= 2^{120/2} = 2^{60} \approx 1.15 \times 10^{18}$.
2. Time $= \dfrac{1.15 \times 10^{18}}{10^{9}} = 1.15 \times 10^{9}$ seconds.
3. A year is about $3.16 \times 10^{7}$ seconds, so $\dfrac{1.15 \times 10^{9}}{3.16 \times 10^{7}} \approx 36.5$ years. The slide rounds to 37. ✓

*Same at 1 THz ($10^{12}$ op/s):* $\dfrac{1.15 \times 10^{18}}{10^{12}} = 1.15 \times 10^{6}$ seconds; a day is $86{,}400$ seconds; $\dfrac{1.15 \times 10^{6}}{86{,}400} \approx 13.3$ days. ✓

*Reached:* 0.6 s, about 37 years, and about 13 days, matching the slide's table. ✓

**The lesson.** A computer 1,000 times faster turned 37 years into 13 days. Still useless. You can't buy your way out of exponential growth, and the sample requirement in high dimensions grows the same way.

---

## 6. The blessing of dimensionality

Here the lecturer flips the story: high dimensions also give you useful things for free. This is his research field (Vector Symbolic Architectures). The slides are pointers rather than full derivations, so these notes explain the ideas in plain words.

### 6.1 Where this sits in AI (slide 39)

Two big camps:

- **Symbolists / GOFAI** ("Good Old-Fashioned AI"): model knowledge with logic and clear rules. Good at symbols and reasoning.
- **Artificial neural networks**: model knowledge by imitating the brain. Criticised by the symbolists for struggling to represent *structured, related* concepts.

In the overlap sit **Vector Symbolic Architectures (VSAs)**, also called hyperdimensional computing. Goal: represent related concepts and do analogy reasoning *using vectors*, so you get the best of both camps.

### 6.2 Kernel transformations (slide 40)

Idea: move the data into a space with **more** dimensions where the similarity landscape is "nicer" for your task. The animation shows red and blue dots mixed on a flat sheet; lift them into a third dimension and a flat plane separates them. One named tool: Random Fourier Features. (Lecture 3 uses this same trick inside the SVM: the kernel trick.)

### 6.3 Separation gets easier (slide 41)

Reference: Gorban and Tyukin, *Stochastic separation theorems*, Neural Networks, 2017. The picture: a circle full of random points, and one point that a single straight line cuts off from all the rest. The theorem says that in high dimensions you can almost always separate a random point from a big random set with a simple linear cut. Good news for classifiers.

### 6.4 Concentration of measure and Lévy's lemma (slide 42)

**Concentration of measure:** in high dimensions, quantities computed from random vectors cluster very tightly around their average. Names attached: Paul Lévy and Vitali Milman.

**Lévy's lemma** for unit vectors $x$ and $y$ (length 1) on a sphere in $d$ dimensions:

$$
P\!\left( \left| \langle x, y \rangle \right| \geq t \right) \;\leq\; 2 \exp\!\left( -\frac{d\, t^2}{2} \right)
$$

Read it slowly:

- $\langle x, y \rangle$ is the dot product. For unit vectors it equals the cosine of the angle between them. Cosine 0 means at right angles, cosine 1 means same direction.
- $P(\ldots \geq t)$ is the probability that the cosine is at least $t$ in size, in other words that the two vectors are *not* roughly perpendicular.
- $d$ is the dimension.
- The right side gets tiny very fast as $d$ grows.

**Why should random vectors be nearly perpendicular at all?**

*Aim:* get a feel for why the lemma is true before plugging in numbers.
*Where we're heading:* showing that a dot product is a long sum of random plus-and-minus terms, which mostly cancel, and that more dimensions means more cancelling.

The dot product adds up $d$ little products, one per dimension:

$$
\langle x, y \rangle = x_1 y_1 + x_2 y_2 + \cdots + x_d y_d .
$$

For random vectors each little product is as likely to be positive as negative, so they mostly cancel, like a long run of coin flips where heads and tails roughly balance. Flip 10 coins and the share of heads wanders widely; flip 10,000 and it sits within about 1% of one half. In the same way, the more dimensions you add, the more the dot product hugs zero. The lemma just puts a number on how tightly.

**Plug in numbers ($t = 0.1$):**

*Aim:* see how quickly the bound collapses as $d$ grows, so "nearly orthogonal" stops being a slogan and becomes a number.

1. $d = 100$: $2 \exp\!\left(-\dfrac{100 \times 0.01}{2}\right) = 2 e^{-0.5} \approx 1.21$. A probability bound above 1 tells you nothing. So in 100 dimensions the lemma gives no guarantee yet.
2. $d = 1{,}000$: $2 e^{-5} \approx 0.0135$, about 1.3%.
3. $d = 10{,}000$: $2 e^{-50} \approx 4 \times 10^{-22}$. Effectively impossible.

**The implication on the slide:** random vectors are nearly orthogonal (nearly perpendicular). **So "symbols" can be encoded as random vectors "for free":** pick a random 10,000-D vector for "cat" and another for "dog", and they are guaranteed to be unrelated. No design work needed.

### 6.5 Quasi-orthogonality with binary vectors (slide 43)

Same story with vectors made of 0s and 1s, $N$ positions long. The **Hamming distance** $\Delta_H(\mathbf{A}, \mathbf{B})$ counts positions where two vectors differ. The **normalised** Hamming distance divides by $N$:

$$
\frac{\Delta_H(\mathbf{A}, \mathbf{B})}{N} \in [0, 1],
$$

so it runs from 0 (identical) to 1 (opposite).

- Two random binary vectors differ in each position with probability one half, so their expected normalised Hamming distance is $\mathbf{0.5}$.
- The plot shows the spread of this distance for $N = 5, 10, 20, 100, 1{,}000, 10{,}000$. Small $N$: wide spread. Large $N$: a thin spike at 0.5.
- "The higher the dimensionality $N$, the more stable is quasi-orthogonality between two random vectors."

Everyday picture: flip 5 coins and the share of heads jumps around. Flip 10,000 and it's almost exactly half every time.

### 6.6 Building structures from vectors (slide 44)

VSAs can represent sequences, trees and graphs **without any learning**, by combining symbol vectors with three operations. (The explanation of the operations is added; the slide just shows the formulas.)

- **$\rho$ (rho)**: a fixed **permutation**, a reshuffle of the vector's positions. Used to mark order or position.
- **$\odot$**: **binding**, element-by-element multiplication. Produces a vector that looks like neither input. Used to glue a role to a value.
- **$+$**: **bundling**, plain addition. Produces a vector similar to all its inputs.

**A toy example with 4-element vectors** (real ones have thousands of elements, but the moves are the same).

*Aim:* actually perform $\rho$, $\odot$ and $+$ once each, so the sequence formula below reads as arithmetic rather than symbols.
*Where we're heading:* three results we can check by eye: a permuted vector, a bound vector that resembles neither parent, and a bundled vector that still resembles both parts.

Let the elements be $+1$ or $-1$.

- $\mathbf{a} = (+1, -1, +1, +1)$, $\mathbf{b} = (-1, -1, +1, -1)$.
- **Permutation $\rho$**: shift everything one place to the right, last element wraps to the front. $\rho(\mathbf{b}) = (-1, -1, -1, +1)$. Same ingredients as $\mathbf{b}$, different order, so it looks like a different symbol: "$\mathbf{b}$, but in position 2".
- **Binding $\odot$**: multiply position by position.
  $$
  \mathbf{a} \odot \mathbf{b} = (+1 \cdot -1,\ -1 \cdot -1,\ +1 \cdot +1,\ +1 \cdot -1) = (-1, +1, +1, -1).
  $$
  Compare with $\mathbf{a}$: it matches $\mathbf{a}$ in 2 of 4 places, which is what pure chance would give. So the bound vector is unrelated to both parents, which is what you want when gluing a role to a value (nobody should mistake "left-child $= \mathbf{a}$" for plain "$\mathbf{a}$").
- **Bundling $+$**: $\mathbf{a} + \rho(\mathbf{b}) = (0, -2, 0, +2)$. Check similarity with $\mathbf{a}$ by dot product: $0 + 2 + 0 + 2 = 4$, clearly positive. With $\rho(\mathbf{b})$: $0 + 2 + 0 + 2 = 4$, also positive. The sum still "remembers" both of its parts. That is how a whole sequence can be stored in one vector and later queried: "is $\mathbf{a}$ in here?" becomes "is the dot product with $\mathbf{a}$ large?"

*Reached:* permutation reorders, binding produces an unrelated vector (2 of 4 matches, chance level), bundling keeps both parts recoverable (dot product 4 with each). ✓

Formulas from the slide:

- **Sequence $a \rightarrow b \rightarrow c$:**
  $$
  \mathbf{s} = \mathbf{a} + \rho(\mathbf{b}) + \rho^{2}(\mathbf{c})
  $$
  First item untouched, second permuted once, third permuted twice. The number of permutations says the position.
- **Tree** ($a$ and $b$ under the left branch, $c$ on the right):
  $$
  \mathbf{s} = \mathbf{r} \odot \mathbf{c} + \mathbf{l} \odot \rho(\mathbf{l} \odot \mathbf{a}) + \mathbf{l} \odot \rho(\mathbf{r} \odot \mathbf{b})
  $$
  $\mathbf{l}$ and $\mathbf{r}$ are vectors for "left" and "right". Each leaf is bound to the path that leads to it.
- **Graph** with edges $a\!\to\!b$, $a\!\to\!c$, $c\!\to\!b$, $c\!\to\!a$:
  $$
  \mathbf{s} = \mathbf{a} \odot \rho(\mathbf{b}) + \mathbf{a} \odot \rho(\mathbf{c}) + \mathbf{c} \odot \rho(\mathbf{b}) + \mathbf{c} \odot \rho(\mathbf{a})
  $$
  Each edge is "source bound with permuted target".

The punchline from the slide: **"Similarity between hypervectors captures similarity between data structures."** Two similar sequences give two similar vectors, so comparing structures becomes comparing vectors with cosine similarity.

### 6.7 Where to read more

hd-computing.com, sites.google.com/view/hdvsaonline, and the mailing list vsacommunity@googlegroups.com.

**Next lecture (per the last slide):** supervised learning, k-nearest neighbours, working with data, at the whiteboard and in Jupyter.

---

## Key points to remember

- Labs: Friday deadline, verbal individual check, no AI-written or copied code. Quizzes: Wed–Fri of theory week, 1.5 h, all four required.
- Six marks of intelligence; the course focuses on the sixth: learning from experience.
- An agent senses through sensors and acts through actuators.
- Learning = pulling general rules out of specific examples, not memorising. Memory still helps.
- Supervised: answers given. Unsupervised: find groups. Reinforcement: only reward/punishment.
- A hypothesis $h$ approximates the hidden function $f$ and must generalise to new data.
- Curse of dimensionality: needed data grows exponentially with features (44 samples in 4-D, $3.8 \times 10^{57}$ in 128-D).
- Exponential algorithms become hopeless fast ($2^{N/2}$: 37 years at $N = 120$; 13 days even 1,000× faster).
- Blessing of dimensionality: random high-dimensional vectors are almost perpendicular (Lévy's lemma; Hamming distance ≈ 0.5), so symbols come "for free" and linear separation gets easier.

## Key terms

- **Agent / sensors / actuators** – the thing that perceives and acts; how it perceives; how it acts.
- **Expert system** – AI built from hand-written if–then rules.
- **Semantic gap** – the missing link between raw sensor data and abstract rules.
- **Training set** – the examples a learner learns from.
- **Hypothesis ($h$)** – the learner's guess at the true function $f$.
- **Generalisation** – getting new, unseen examples right.
- **Clustering** – grouping data without labels.
- **Interpolation / extrapolation** – predicting inside / outside the data you have.
- **Dimension** – number of features.
- **Curse of dimensionality** – needed samples grow exponentially with dimension.
- **NP-complete** – problems whose known solutions take exponential time.
- **Concentration of measure** – in high dimensions random quantities cluster near their mean.
- **Orthogonal / quasi-orthogonal** – at right angles / almost at right angles.
- **Hamming distance** – positions where two binary vectors differ.
- **VSA / hyperdimensional computing** – computing with high-dimensional random vectors; permutation $\rho$, binding $\odot$, bundling $+$.

## Practice questions

**1. Name the three main types of learning and give the signal each one gets.**
Supervised: the correct answer. Unsupervised: nothing, just the data. Reinforcement: a reward or punishment from the environment.

**2. What does it mean for a hypothesis to generalise well?**
It predicts the right output for new examples that were not in the training set.

**3. Why are 100 samples in 50 dimensions "nothing specific"?**
Because the number of samples needed grows exponentially with dimension; 100 points are far too few to show any structure in a 50-D space.

**4. The exponential algorithm takes 1100 s at $N = 80$ on a 1 GHz machine. Roughly how long at $N = 82$?**
Each extra 2 bits doubles $2^{N/2}$, so about $2{,}200$ s.

**5. Two random unit vectors in 10,000 dimensions: roughly what is their cosine, and why?**
Close to 0 (nearly perpendicular), because by Lévy's lemma the chance of a cosine of even $0.1$ is about $4 \times 10^{-22}$.

**6. In the sequence formula $\mathbf{s} = \mathbf{a} + \rho(\mathbf{b}) + \rho^{2}(\mathbf{c})$, what does $\rho^{2}$ mean and why is it there?**
Apply the permutation twice. It marks $\mathbf{c}$ as the third element; the number of permutations encodes position.
