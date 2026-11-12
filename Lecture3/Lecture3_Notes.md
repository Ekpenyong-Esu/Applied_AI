# Lecture 3 Notes – Support Vector Machines (the whiteboard lecture)

**Course:** D7041E Applied Artificial Intelligence
**Source:** `L3-svm.pdf`, 10 handwritten whiteboard pages. Everything here was written by the teacher by hand, so all of it counts as important. Hard-to-read parts are marked [unclear: my best guess].

**One reading tip before you start:** on these pages the teacher writes zero as **φ** (it looks like ∅). Every "$= \varphi$" means "$= 0$".

**What this lecture covers:**

1. The setting: a model that maps data to labels.
2. Preliminaries: lines, hyperplanes, and the distance from a point to a line.
3. Lagrangian optimisation (minimising with a constraint), with a worked example.
4. The geometric view: two classes, labels $+1$ and $-1$, support vectors.
5. The margin, and turning "widest margin" into something we can minimise.
6. The optimisation problem, written properly.
7. The Lagrangian and the dual problem, where the kernel appears.
8. Solving it: support vectors, computing $w$ and $b$.
9. Using SVM for more than two classes.
10. Linear vs non-linear separability, and the lifting trick.

---

## How every derivation in these notes is laid out

Each derivation follows the same four moves, so you always know why you are doing a step:

1. **Aim** – the question we are trying to answer.
2. **Where we're heading** – what the final formula should look like, stated before any algebra.
3. **Steps** – numbered, each with the reason it is done.
4. **Reached** – a one-line check that we arrived where we said we would.

If you get lost in the middle of a derivation, go back to its "Where we're heading" line.

---

## The story in one paragraph

k-NN (Lecture 2) remembers everything and compares. An SVM instead *draws a line* between two classes and keeps only the line. But which line? The one with the **widest empty gap** on both sides. "Widest gap" turns into a maths problem: minimise $\tfrac{1}{2}\lVert w \rVert^2$ subject to every point being on the right side with room to spare. We solve it with Lagrange multipliers. The solution has a surprise: only a few points matter (the support vectors), and the data only ever appear as dot products, which lets us swap in a **kernel** and draw curved boundaries.

---

## 1. The setting (page 1)

Data $\mathbf{X} = \{x_i\}$ goes into a box called **Model**, which outputs a category. The teacher writes "Cat $i$" then **"rename to $y_i$"**, and:

$$
y_i = f(x)
$$

So the model is a function $f$ from a feature vector to a label. Same idea as the hypothesis $h$ in Lecture 1. The difference from k-NN: an SVM *learns* $f$ and then forgets most of the data.

---

## 2. Preliminaries: lines and distances (page 2)

### A line, two ways

**School way:** $x_2 = \alpha x_1 + \beta$. $\alpha$ (alpha) is the slope (the angle drawn at the line), $\beta$ (beta) is where the line crosses the $x_2$ axis.

**General way:**

$$
a_1 x_1 + a_2 x_2 + b = 0
$$

Why bother? Because the general way works for any number of features:

$$
\sum_i a_i x_i + b = 0 \qquad \text{or, with vectors,} \qquad \mathbf{a} \cdot \mathbf{x} + b = 0
$$

Here $\mathbf{x} = \{x_1, x_2, \ldots, x_i, \ldots\}$ is the feature vector (the teacher writes "vector" under it) and $\mathbf{a}$ is a vector of coefficients. In 2-D this is a line, in 3-D a flat plane, in more dimensions it's called a **hyperplane**. (Added: the vector $\mathbf{a}$ points straight out of the hyperplane at a right angle.)

**Switching between forms, worked example.** Take $2x_1 + x_2 - 4 = 0$ ($a_1 = 2$, $a_2 = 1$, $b = -4$).

1. Move everything but $x_2$ to the right: $x_2 = -2x_1 + 4$.
2. So $\alpha = -a_1/a_2 = -2$ and $\beta = -b/a_2 = 4$.

### Distance from a point to a line (right-hand sketch)

Point $P(x_1^{\circ}, x_2^{\circ})$ and the line $a_1 x_1 + a_2 x_2 + b = 0$. The distance $d$, drawn as a short perpendicular from $P$ to the line, is

$$
d = \frac{\left| a_1 x_1^{\circ} + a_2 x_2^{\circ} + b \right|}{\sqrt{a_1^2 + a_2^2}}
$$

How to use it: plug the point's coordinates into the left side of the line equation (this gives 0 only if the point sits *on* the line), take the absolute value, divide by the length of $\mathbf{a}$. The division turns the raw number into a true perpendicular distance.

**Where the formula comes from, the easy way.**

*Aim:* understand why the distance is "plug the point into the line equation, then divide by $\sqrt{a_1^2 + a_2^2}$", rather than just memorise it.
*Where we're heading:* we will show that the top of the fraction gives the right answer for a simple line but changes if you rewrite the same line with bigger coefficients, and that the bottom of the fraction exactly cancels that change.

Start with a line you can see by eye: the horizontal line $x_2 = 2$. In general form that is $0 \cdot x_1 + 1 \cdot x_2 - 2 = 0$, so $a_1 = 0$, $a_2 = 1$, $b = -2$. Take the point $(5, 7)$. Its distance to the line is obviously $7 - 2 = 5$ (just count up from height 2 to height 7). The formula agrees: top $= |0 + 7 - 2| = 5$, bottom $= \sqrt{0 + 1} = 1$, $d = 5$. Now here is the catch. The *same* line can be written $0 \cdot x_1 + 2 \cdot x_2 - 4 = 0$ (everything doubled). The top now gives $|14 - 4| = 10$, which is wrong by a factor of 2. But the bottom also doubles: $\sqrt{0 + 4} = 2$, and $10/2 = 5$ again. That is the whole job of the denominator: the top number grows with how big you wrote the coefficients, and dividing by the length of $\mathbf{a}$ cancels that out, leaving the real distance. For a tilted line the same thing happens, only with Pythagoras hidden in the $\sqrt{a_1^2 + a_2^2}$.

*Reached:* the numerator measures "how far off the line", the denominator removes the arbitrary scale of the coefficients, and together they give the true perpendicular distance. ✓

**Worked example.** $P = (3, 3)$, line $2x_1 + x_2 - 4 = 0$.

1. Top: $2 \times 3 + 1 \times 3 - 4 = 5$. Absolute value $5$.
2. Bottom: $\sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.236$.
3. $d = 5 / 2.236 \approx \mathbf{2.236}$.

Check: $(2, 0)$ is on the line ($2 \times 2 + 0 - 4 = 0$), so its distance is 0. ✓

Keep this formula in your pocket. The whole SVM is built on it.

---

## 3. Lagrangian optimisation (page 3)

**The problem type.** You have a function $f(x)$ and a constraint $g(x)$. **Minimise $f(x)$** while obeying $g$. The sketch shows a wiggly $f(x)$ with several dips; the constraint says which part of the $x$-axis you're allowed to use.

**The tool.** Combine them into one function with a new unknown $\alpha$ (alpha), the **Lagrange multiplier**:

$$
L(x, \alpha) = f(x) - \alpha \, g(x)
$$

where $g$ must be written in the form $g(x) = 0$. The constrained minimum is where **every partial derivative of $L$ is zero**. (Added: that's the standard rule; the page only writes the Lagrangian.)

Picture: you want the lowest point in a valley (minimise $f$), but you must stay on a marked trail (the constraint). $\alpha$ measures how hard the trail pulls you away from where you'd otherwise go.

**Why does "set the derivatives of $L$ to zero" find the answer?**

*Aim:* see why one recipe (build $L = f - \alpha g$, set all its partial derivatives to zero) solves a minimisation that has a side condition.
*Where we're heading:* we want to end up with two facts: (1) at the best point the slope of $f$ is a multiple of the slope of $g$, and (2) the point still satisfies $g = 0$. We will show that the derivatives of $L$ say exactly these two things.

Walk along the trail. As long as the ground slopes *along* the trail, you can keep walking and go lower. You stop at the point where the slope of $f$ points straight *across* the trail, so walking along it no longer helps. "Slope of $f$ points across the trail" is the same as "slope of $f$ is some multiple of the slope of $g$", written

$$
\nabla f = \alpha \nabla g \qquad \Longleftrightarrow \qquad \nabla f - \alpha \nabla g = 0 .
$$

That is exactly $\partial L / \partial x = 0$ for $L = f - \alpha g$. And $\partial L / \partial \alpha = 0$ just says $-g = 0$, which keeps you on the trail. So the two kinds of derivative express the two things you need: "no improvement left along the trail" and "still on the trail".

Tiny check with numbers: $f = x^2$ (a bowl with its bottom at $x = 0$) subject to $x = 3$. $L = x^2 - \alpha(x - 3)$. $\partial L/\partial x = 2x - \alpha = 0$ gives $\alpha = 2x$; $\partial L/\partial \alpha = 0$ gives $x = 3$. So $x = 3$ and $\alpha = 6$. The answer is forced to the trail ($x = 3$), and $\alpha = 6$ is the slope of the bowl there, telling you how hard the constraint is holding you away from the free minimum at 0.

*Reached:* $\partial L/\partial x = 0$ gave "slope of $f$ = $\alpha$ × slope of $g$", and $\partial L/\partial \alpha = 0$ gave "$g = 0$". Both facts we were heading for, from one recipe. ✓

> **⚠ Check with your teacher.** The page reads "min $f$: $2 + x^2 + 2y^2 = 0$, such that $g$: $x + y = 1$". The "$= 0$" after $f$ can't be right: $2 + x^2 + 2y^2$ is always at least 2. The intended task is **minimise $f = 2 + x^2 + 2y^2$ subject to $x + y = 1$**. Also, the constraint must be rearranged to $g = x + y - 1 = 0$ before going into $L$. The solution below uses that reading.

**Worked solution.**

*Aim:* find the point $(x, y)$ on the line $x + y = 1$ where $f = 2 + x^2 + 2y^2$ is smallest, and the value of $f$ there.
*Where we're heading:* three unknowns ($x$, $y$, $\alpha$) and three equations (one partial derivative each). Solving them should hand us specific numbers for $x$, $y$ and $\alpha$; plugging $x$ and $y$ back into $f$ gives the minimum.

1. Constraint as $g = 0$: $g(x, y) = x + y - 1$.
2. Lagrangian:
   $$
   L(x, y, \alpha) = 2 + x^2 + 2y^2 - \alpha (x + y - 1)
   $$
3. $\dfrac{\partial L}{\partial x} = 2x - \alpha = 0 \;\Rightarrow\; \mathbf{x = \alpha/2}$. (At the constrained minimum, $L$ is flat in the $x$ direction.)
4. $\dfrac{\partial L}{\partial y} = 4y - \alpha = 0 \;\Rightarrow\; \mathbf{y = \alpha/4}$.
5. $\dfrac{\partial L}{\partial \alpha} = -(x + y - 1) = 0 \;\Rightarrow\; \mathbf{x + y = 1}$. (Differentiating with respect to $\alpha$ just hands you the constraint back. That's the trick.)
6. Put 3 and 4 into 5: $\dfrac{\alpha}{2} + \dfrac{\alpha}{4} = 1 \;\Rightarrow\; \dfrac{3\alpha}{4} = 1 \;\Rightarrow\; \mathbf{\alpha = 4/3}$.
7. So $x = 2/3$, $y = 1/3$. Check: $2/3 + 1/3 = 1$. ✓
8. Minimum value: $f = 2 + \dfrac{4}{9} + 2 \cdot \dfrac{1}{9} = 2 + \dfrac{6}{9} = \mathbf{2\tfrac{2}{3} \approx 2.667}$.

Sanity check: the naive guess $x = y = 0.5$ gives $2 + 0.25 + 0.5 = 2.75$, which is worse. The $y^2$ term has a 2 in front, so it pays to keep $y$ small. ✓

*Reached:* the constrained minimum is at $(x, y) = (2/3, 1/3)$ with $f = 8/3$, and $\alpha = 4/3$. Exactly the three numbers plus the minimum value we set out to find. ✓

---

## 4. Geometric view of the data (page 4)

Two groups of points in the $x_1$–$x_2$ plane: circles (o) and crosses (+). Each sample $i$ gets a label

$$
y_i =
\begin{cases}
+1 & \text{"positive"} \\
-1 & \text{"negative"}
\end{cases}
$$

"label of category $i$". Using $+1$ and $-1$ instead of names is on purpose; it makes the algebra below work.

Between the groups the teacher draws **three parallel lines**: a middle one and two outer ones that just touch the nearest points of each class. Those touching points are boxed and labelled **support vectors**. They "hold up" the boundary. Move any other point a little and nothing changes; move a support vector and the boundary moves.

**Change of notation:** $a_i \rightarrow w_i$. So $\mathbf{a}\cdot\mathbf{x} + b = 0$ becomes

$$
\mathbf{w} \cdot \mathbf{x} + b = 0
$$

($w$ for weights.)

**The constraints** ("for any sample $i$"):

$$
\begin{aligned}
\mathbf{w} \cdot \mathbf{x}_i + b &\geq +1 \quad \text{for } y_i = +1 \\
\mathbf{w} \cdot \mathbf{x}_i + b &\leq -1 \quad \text{for } y_i = -1
\end{aligned}
$$

Positive points must be on or beyond the line $\mathbf{w}\cdot\mathbf{x} + b = +1$; negative points on or beyond $\mathbf{w}\cdot\mathbf{x} + b = -1$. The strip between $-1$ and $+1$ must stay empty.

---

## 5. The margin, and what to optimise (page 5)

The sketch: three lines, $\mathbf{w}\cdot\mathbf{x} + b = -1$, $\mathbf{w}\cdot\mathbf{x} + b = 0$ (thick, labelled **"target model"**) and $\mathbf{w}\cdot\mathbf{x} + b = +1$. The distance from the middle line to one outer line is $d$; the full width of the empty strip is $D = 2d$.

Use the point-to-line formula from section 2 with $\mathbf{w}$ in place of $\mathbf{a}$:

$$
d = \frac{\left| w_1 x_1^{\circ} + w_2 x_2^{\circ} + b \right|}{\sqrt{w_1^2 + w_2^2}} = \frac{\left| \mathbf{w}\cdot\mathbf{x} + b \right|}{\lVert \mathbf{w} \rVert}
$$

$\lVert \mathbf{w} \rVert$ is the length of $\mathbf{w}$. For a support vector, $|\mathbf{w}\cdot\mathbf{x} + b| = 1$ (it sits on an outer line), so

$$
d = \frac{1}{\lVert \mathbf{w} \rVert} \qquad\qquad D = 2d = \frac{2}{\lVert \mathbf{w} \rVert}
$$

**"Function to optimise."**

*Aim:* turn the wish "make the empty strip as wide as possible" into a clean mathematical problem that a standard optimiser can solve.
*Where we're heading:* a **minimisation** of a smooth, bowl-shaped function of $\mathbf{w}$ alone. We will get there by rewriting "maximise $D$" twice.

We want the widest strip: **maximise $D = 2/\lVert \mathbf{w} \rVert$**. The teacher reshapes this in two steps:

1. Maximising $D$ is the same as **minimising** $\dfrac{1}{D} = \dfrac{1}{2}\lVert \mathbf{w} \rVert$. (A fraction gets big when its reciprocal gets small.)
2. **"Make convex":** minimise
   $$
   \frac{1}{2} \lVert \mathbf{w} \rVert^2
   $$
   instead. Squaring doesn't move the minimum ($\lVert \mathbf{w} \rVert$ is never negative), but $\lVert \mathbf{w} \rVert^2$ is a smooth bowl, which optimisers love, while $\lVert \mathbf{w} \rVert$ has a sharp kink at zero. The $\tfrac{1}{2}$ is kept so the derivative comes out as plain $\mathbf{w}$.

*Reached:* "widest strip" has become "minimise $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2$", a convex function of $\mathbf{w}$. That is the objective used in the rest of the lecture. ✓

Why want a wide strip at all? A boundary squeezed against the data breaks as soon as a new point is slightly off. The widest strip is the safest bet.

**Worked example** (added). Say the solution is $\mathbf{w} = (1, 1)$, $b = -3$, with training points $(1,1)$ labelled $-1$, $(2,2)$ labelled $+1$, $(0.5,1)$ labelled $-1$, $(3,2)$ labelled $+1$.

1. Check each point with $y_i(\mathbf{w}\cdot\mathbf{x}_i + b)$:
   - $(1,1)$: $1 + 1 - 3 = -1$; times $-1$ gives $\mathbf{1}$. On the outer line: a support vector.
   - $(2,2)$: $2 + 2 - 3 = +1$; times $+1$ gives $\mathbf{1}$. A support vector.
   - $(0.5,1)$: $0.5 + 1 - 3 = -1.5$; times $-1$ gives $1.5 \geq 1$. ✓ Not a support vector.
   - $(3,2)$: $3 + 2 - 3 = 2$; times $+1$ gives $2 \geq 1$. ✓
2. $\lVert \mathbf{w} \rVert = \sqrt{1 + 1} = \sqrt{2} \approx 1.414$.
3. $d = 1/\sqrt{2} \approx 0.707$; $D = 2/\sqrt{2} \approx 1.414$.
4. Objective $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2 = \tfrac{1}{2} \times 2 = 1$.

---

## 6. The optimisation formulation (page 6)

$$
\begin{aligned}
\text{minimise} \quad & f(\mathbf{w}) = \tfrac{1}{2}\lVert \mathbf{w} \rVert^2 \\
\text{subject to} \quad & \mathbf{w}\cdot\mathbf{x}_i + b \geq +1 \quad \text{for } y_i = +1 \\
& \mathbf{w}\cdot\mathbf{x}_i + b \leq -1 \quad \text{for } y_i = -1
\end{aligned}
$$

**Merge the two constraints into one.**

*Aim:* replace two separate rules (one for positive points, one for negative) with a single rule, so that later we can attach one multiplier $\alpha_i$ per point without caring which class it is in.
*Where we're heading:* one inequality of the form $y_i(\ldots) \geq 1$ that holds for every sample.

Multiply each by its own $y_i$. For $y_i = +1$ nothing changes. For $y_i = -1$, multiplying "$\leq -1$" by $-1$ flips it to "$\geq +1$". Both become:

$$
y_i \left( \mathbf{w}\cdot\mathbf{x}_i + b \right) \geq 1 \qquad \text{for every } i
$$

(Check with the example above: every $y_i(\mathbf{w}\cdot\mathbf{x}_i + b)$ was $\geq 1$. ✓)

*Reached:* one constraint for all samples, which is what the Lagrangian below needs. ✓

**Write the constraint as $g = 0$** (the Lagrangian wants that form). For the support vectors the inequality is an equality, $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) = 1$, so

$$
g(\mathbf{w}) : \quad y_i \left( \mathbf{w}\cdot\mathbf{x}_i + b \right) - 1 = 0
$$

**The Lagrangian**, one multiplier $\alpha_i$ per sample ("for all samples"):

$$
L(\mathbf{w}, \alpha) = f(\mathbf{w}) - \sum_i \alpha_i \, g(\mathbf{w}, \mathbf{x}_i)
$$

---

## 7. The Lagrangian and the dual problem (page 7)

Written out:

$$
L = \frac{1}{2}\lVert \mathbf{w} \rVert^2 - \sum_i \alpha_i \Big[ y_i \left( \mathbf{w}\cdot\mathbf{x}_i + b \right) - 1 \Big]
\qquad \text{(sum over all samples)}
$$

**"Formulate dual optimisation problem."**

*Aim:* get rid of $\mathbf{w}$ and $b$ entirely, so that the problem is written only in terms of the multipliers $\alpha_i$.
*Where we're heading:* a new function $L^{*}(\alpha)$ in which the training points appear **only inside dot products** $\mathbf{x}_i \cdot \mathbf{x}_j$. That last property is the whole prize: it is what later lets us swap the dot product for a kernel. Keep an eye out for the moment the dot products appear.

The original (primal) problem has unknowns $\mathbf{w}$ and $b$. The dual rewrites everything so only the $\alpha_i$ remain. Steps:

**Step 1. Derivative with respect to $\mathbf{w}$, set to zero.**

$$
\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{x}_i = 0
\qquad \Longrightarrow \qquad
\mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i
$$

($n$ = number of samples.) Why: the derivative of $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2$ is $\mathbf{w}$; the derivative of $-\alpha_i y_i (\mathbf{w}\cdot\mathbf{x}_i)$ with respect to $\mathbf{w}$ is $-\alpha_i y_i \mathbf{x}_i$. Meaning: **the weight vector is a weighted sum of the training points.**

*Seeing those two derivatives in 2-D.* With $\mathbf{w} = (w_1, w_2)$,

$$
\tfrac{1}{2}\lVert \mathbf{w} \rVert^2 = \tfrac{1}{2}(w_1^2 + w_2^2),
\qquad
\frac{\partial}{\partial w_1} = \tfrac{1}{2} \cdot 2 w_1 = w_1,
\qquad
\frac{\partial}{\partial w_2} = w_2 .
$$

Stack them and you get $(w_1, w_2) = \mathbf{w}$. That is why the $\tfrac{1}{2}$ was kept: it eats the 2 from the power rule. For the other term, $\mathbf{w}\cdot\mathbf{x}_i = w_1 x_{i1} + w_2 x_{i2}$; its derivative with respect to $w_1$ is $x_{i1}$ and with respect to $w_2$ is $x_{i2}$, so the derivative of $\mathbf{w}\cdot\mathbf{x}_i$ is just $\mathbf{x}_i$, and the constants $-\alpha_i y_i$ ride along.

*What "weighted sum of training points" means with numbers.* Suppose only two points have non-zero $\alpha$: $\mathbf{x}_1 = (1, 1)$ with $y_1 = -1$, $\alpha_1 = 0.5$, and $\mathbf{x}_2 = (2, 2)$ with $y_2 = +1$, $\alpha_2 = 0.5$. Then

$$
\mathbf{w} = 0.5 \cdot (-1) \cdot (1, 1) + 0.5 \cdot (+1) \cdot (2, 2) = (-0.5, -0.5) + (1, 1) = (0.5, 0.5).
$$

The boundary direction is built out of the data points themselves, which is why only the support vectors are needed at the end.

**Step 2. Derivative with respect to $b$, set to zero.**

$$
\frac{\partial L}{\partial b} = -\sum_i \alpha_i y_i = 0
\qquad \Longrightarrow \qquad
\sum_i \alpha_i y_i = 0
$$

**Step 3. Substitute back into $L$.** (Added: the in-between algebra.) First expand $L$:

$$
L = \tfrac{1}{2}\, \mathbf{w}\cdot\mathbf{w} \;-\; \sum_i \alpha_i y_i (\mathbf{w}\cdot\mathbf{x}_i) \;-\; b \sum_i \alpha_i y_i \;+\; \sum_i \alpha_i .
$$

- The term $b \sum_i \alpha_i y_i$ is zero by Step 2.
- With $\mathbf{w} = \sum_j \alpha_j y_j \mathbf{x}_j$:
  $$
  \tfrac{1}{2}\, \mathbf{w}\cdot\mathbf{w} = \tfrac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j).
  $$
- And
  $$
  \sum_i \alpha_i y_i (\mathbf{w}\cdot\mathbf{x}_i) = \sum_i \sum_j \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j),
  $$
  the same double sum without the $\tfrac{1}{2}$.
- $\tfrac{1}{2}(\text{double sum}) - (\text{double sum}) = -\tfrac{1}{2}(\text{double sum})$.

**Step 4. Result, "Lagrangian for dual problem":**

$$
L^{*} = \sum_{i=1}^{n} \alpha_i \;-\; \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i y_i \, \alpha_j y_j \, (\mathbf{x}_i \cdot \mathbf{x}_j)
$$

Maximise this over the $\alpha_i$, with $\alpha_i \geq 0$ and $\sum_i \alpha_i y_i = 0$. (Added: $\alpha_i \geq 0$ is standard; the page shows only the second condition.)

*Reached:* $\mathbf{w}$ and $b$ are gone, only the $\alpha_i$ remain, and the data show up solely as $(\mathbf{x}_i \cdot \mathbf{x}_j)$. Both things we were heading for. ✓

**The big observation.** The teacher draws an arrow labelled **"kernel"** to the bracket $(\mathbf{x}_i \cdot \mathbf{x}_j)$ and writes $K(\mathbf{x}_i, \mathbf{x}_j)$. The data only ever appear as **dot products between pairs of samples**. So if you replace the dot product with some other similarity function $K$, called a **kernel**,

$$
(\mathbf{x}_i \cdot \mathbf{x}_j) \;\longrightarrow\; K(\mathbf{x}_i, \mathbf{x}_j),
$$

the whole method still works, but now it behaves as if the data had been moved into a higher-dimensional space (Lecture 1, slide 40). That is the **kernel trick**, and it's why the lab asks for linear, polynomial and RBF kernels.

---

## 8. Solving: support vectors, $\mathbf{w}$ and $b$ (page 8)

1. **"Solve $L^{*}$ w.r.t. $\alpha_i$."** This is a quadratic optimisation problem; software (scikit-learn in the lab) does it for you.
2. **"As a result only few $\alpha_i > 0$."** Most multipliers come out exactly zero. **"We take these samples $\mathbf{x}_i$ s.t. $\alpha_i > 0$."** They are the **support vectors**. Every other training point can be thrown away.
3. **"Compute model $M$: $\mathbf{w}\cdot\mathbf{x} + b$"** with
   $$
   \mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i
   $$
   Only the support vectors contribute, because the other $\alpha_i$ are zero.
4. **"For computing $b$":**
   *Aim:* the dual gave us $\mathbf{w}$ but said nothing about $b$, so we need a separate way to get it.
   *Where we're heading:* a formula of the form $b = (\text{something we know})$, using only a support vector, its label and the $\mathbf{w}$ we already have.
   Take any support vector (where the constraint is an equality).
   - Start: $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) = 1$.
   - Multiply both sides by $y_i$: $y_i \, y_i \,(\mathbf{w}\cdot\mathbf{x}_i + b) = y_i$. (The teacher boxes "$y_i y_i$".)
   - $y_i$ is $+1$ or $-1$, so $y_i y_i = 1$. Hence $\mathbf{w}\cdot\mathbf{x}_i + b = y_i$.
   - Therefore
     $$
     b = y_i - \mathbf{w}\cdot\mathbf{x}_i
     $$
   The arrow on the page feeds this $b$ back into the model.
   *Reached:* $b$ expressed through known quantities only ($y_i$, $\mathbf{w}$, $\mathbf{x}_i$). ✓

**Worked example.** $\mathbf{w} = (1, 1)$ from section 5. Support vector $(2, 2)$, $y = +1$: $b = 1 - (2 + 2) = \mathbf{-3}$. Other support vector $(1, 1)$, $y = -1$: $b = -1 - (1 + 1) = \mathbf{-3}$. Same answer, as it should be. ✓

**Classifying a new point** (added, standard): compute $\mathbf{w}\cdot\mathbf{x} + b$; positive → class $+1$, negative → class $-1$. In one line,

$$
\hat{y} = \operatorname{sign}\left( \mathbf{w}\cdot\mathbf{x} + b \right).
$$

Example: $(0, 0)$ gives $-3$ → class $-1$.

---

## 9. Practical usage: more than two classes (page 9)

An SVM separates two classes. "How to use SVM in multi-class classification tasks": two strategies.

**One vs. one.** One SVM for every pair: $cl_1$ vs $cl_2$, $cl_1$ vs $cl_3$, … For $m$ classes:

$$
\frac{m\,(m - 1)}{2} \ \text{SVMs}
$$

To classify a sample, run all of them and count votes; the class with most votes wins (added: the voting rule).

Why that formula? Each of $m$ classes pairs with $m - 1$ others, giving $m(m-1)$ pairs, but "cat vs dog" is the same machine as "dog vs cat", so halve it.

Counts: $m = 3 \rightarrow 3$; $m = 4 \rightarrow 6$; $m = 10$ (the digits) $\rightarrow 45$.

**One vs. all** (also called one vs. rest). "$m$ classes $= M$ SVMs" [unclear: meaning $m$ SVMs]. SVM number $i$ separates class $i$ from everyone else. The sketch shows a table with columns $cl_1, cl_2, \ldots$ and a column of scores, captioned "class for a given sample": you give the sample the class whose SVM is most confident.

---

## 10. Linear vs non-linear separability (page 10)

**Left sketch:** crosses and circles that one straight line splits cleanly. Linearly separable. A plain SVM is fine.

**Right sketch:** circles in the middle, crosses in a ring around them. No straight line works. Not linearly separable in $(x_1, x_2)$.

**Bottom sketch, the fix:** add a new axis

$$
z = x_1^2 + x_2^2 ,
$$

the squared distance from the centre. Inner points get a small value, ring points a large one. In this lifted 3-D view the two groups sit at different heights and a flat plane separates them. The kernel in section 7 does this lifting for you, without ever computing the new coordinates.

**Worked example.** Inner circles at $(0.5, 0.5)$ and $(-0.6, 0.3)$; outer crosses at $(2, 1)$ and $(-1.5, -1.5)$.

1. $(0.5, 0.5) \rightarrow 0.25 + 0.25 = 0.5$.
2. $(-0.6, 0.3) \rightarrow 0.36 + 0.09 = 0.45$.
3. $(2, 1) \rightarrow 4 + 1 = 5$.
4. $(-1.5, -1.5) \rightarrow 2.25 + 2.25 = 4.5$.
5. Rule "$z < 2 \rightarrow$ circle, else cross" separates them perfectly, with a straight cut in the new coordinate.

---

## Key points to remember

- Hyperplane: $\mathbf{w}\cdot\mathbf{x} + b = 0$. Distance from a point to it: $|\mathbf{w}\cdot\mathbf{x} + b| / \lVert \mathbf{w} \rVert$.
- Labels are $+1$ and $-1$. Constraint: $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \geq 1$ for all $i$.
- Margin width $D = 2/\lVert \mathbf{w} \rVert$. Maximising it = minimising $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2$ (convex).
- Lagrangian: $L = \tfrac{1}{2}\lVert \mathbf{w} \rVert^2 - \sum_i \alpha_i[y_i(\mathbf{w}\cdot\mathbf{x}_i + b) - 1]$. Setting derivatives to zero gives $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ and $\sum_i \alpha_i y_i = 0$.
- Dual: $L^{*} = \sum_i \alpha_i - \tfrac{1}{2}\sum_i\sum_j \alpha_i \alpha_j y_i y_j (\mathbf{x}_i\cdot\mathbf{x}_j)$. Data appear only as dot products → swap in a kernel $K$.
- Only samples with $\alpha_i > 0$ matter: the support vectors. $b = y_i - \mathbf{w}\cdot\mathbf{x}_i$ from any of them.
- Multi-class: one vs one needs $m(m-1)/2$ SVMs; one vs all needs $m$.
- Non-separable data become separable after lifting to more dimensions; the kernel does the lifting.
- Lagrange recipe: write the constraint as $g = 0$, form $L = f - \alpha g$, set all partial derivatives to zero, solve.

## Key terms

- **Hyperplane** – the flat boundary $\mathbf{w}\cdot\mathbf{x} + b = 0$.
- **Norm $\lVert \mathbf{w} \rVert$** – the length of $\mathbf{w}$.
- **Margin** – the empty strip between the classes, width $2/\lVert \mathbf{w} \rVert$.
- **Support vectors** – training points on the edge of the strip; the only ones with $\alpha_i > 0$.
- **Lagrangian / Lagrange multiplier $\alpha$** – the way to minimise under a constraint.
- **Primal / dual** – the problem in $\mathbf{w}$ and $b$ / the same problem in the $\alpha_i$.
- **Convex** – bowl-shaped, one minimum, easy to optimise.
- **Kernel $K(\mathbf{x}_i, \mathbf{x}_j)$** – a similarity function that replaces the dot product.
- **Kernel trick** – using a kernel to act as if in a higher-dimensional space without going there.
- **One vs one / one vs all** – pairwise SVMs / one SVM per class.
- **Linearly separable** – splittable by a straight hyperplane.

## Practice questions

**1. Distance from $(1, 2)$ to the line $3x_1 + 4x_2 - 1 = 0$?**
Top: $3 + 8 - 1 = 10$. Bottom: $\sqrt{9 + 16} = 5$. $d = 2$.

**2. $\mathbf{w} = (3, 4)$. Margin width? If $(1, 1)$ is a support vector with label $+1$, what is $b$?**
$\lVert \mathbf{w} \rVert = 5$, $D = 2/5 = 0.4$. $b = 1 - (3 + 4) = -6$.

**3. Minimise $f = x^2 + y^2$ subject to $x + 2y = 5$.**
$g = x + 2y - 5$. $L = x^2 + y^2 - \alpha(x + 2y - 5)$. $\partial/\partial x$: $2x = \alpha \Rightarrow x = \alpha/2$. $\partial/\partial y$: $2y = 2\alpha \Rightarrow y = \alpha$. Constraint: $\alpha/2 + 2\alpha = 5 \Rightarrow \alpha = 2$. So $x = 1$, $y = 2$, $f = 5$.

**4. Why do we minimise $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2$ rather than maximise $2/\lVert \mathbf{w} \rVert$ directly?**
Same solution, but $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2$ is smooth and convex, so standard optimisers handle it reliably.

**5. Why does the kernel trick work for SVM specifically?**
Because in the dual the training data appear only through dot products $\mathbf{x}_i\cdot\mathbf{x}_j$, which can be replaced by any kernel $K(\mathbf{x}_i, \mathbf{x}_j)$.

**6. Six classes: how many SVMs with one vs one, and with one vs all?**
$15$ and $6$.

**7. A point has $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) = 3.2$. Is it a support vector?**
No. Support vectors have exactly $1$. This point is comfortably inside its own side.
