# 拉格朗日对偶问题的数学直觉

## 对偶问题的导出

### 数理逻辑视角

拉格朗日对偶所求解的问题如下：
$$\min \{f(\mathbf{x}): g_i(\mathbf{x})\leq 0,\; i=1,2,...,m\}$$

若方程组：
$$
\begin{aligned}
&f(\mathbf{x})<v \\
&g_{i}(\mathbf{x}) \leq 0, i=1,2, \ldots, m
\end{aligned}
$$
无解，则我们可以得到v为f的一个下界。

同时，我们知道，若方程组有解，则对$\bm{\forall \lambda}\geq \mathbf{0}$,使得以下方程成立。

$$f(\mathbf{x})+\sum_{i=1}^{m}\lambda_ig_i(\mathbf{x})<v$$

故根据数理逻辑的角度出发，逆否命题：若$\bm{\exist \lambda}\geq \mathbf{0}$,使得：

$$\min_{\mathbf{x}} f(\mathbf{x})+\sum_{i=1}^m \lambda_ig_i(\mathbf{x})\geq v$$

成立，则可推导出原方程组无解，v为f的下界。

故原先的优化目标变为寻找最优下界v：

$$v=\max_{\bm{\lambda}\geq \mathbf{0}}\min_{\mathbf{x}} f(\mathbf{x})+\sum_{i=1}^{m}\lambda_ig_i(\mathbf{x})$$

当满足KKT条件，则v成为f的下确界。

### 博弈论视角

我们需要证明在拉格朗日对偶问题中：

$$\max_{\bm{\lambda}\geq \mathbf{0}}\min_{\mathbf{x}} f(\mathbf{x})+\sum_{i=1}^{m}\lambda_ig_i(\mathbf{x})\leq \min_{\mathbf{x}} \max_{\bm{\lambda}\geq \mathbf{0}} f(\mathbf{x})+\sum_{i=1}^{m}\lambda_ig_i(\mathbf{x})$$

从博弈论入手：

$$
\begin{aligned}
&\text { Define } g(z) \triangleq \inf _{w \in W} f(z, w) \\
&\forall w \in W, \forall z \in Z, g(z) \leq f(z, w) \\
&\Longrightarrow \forall w \in W, \sup _{z \in Z} g(z) \leq \sup _{z \in Z} f(z, w) \\
&\Longrightarrow \sup _{z \in Z} g(z) \leq \inf _{w \in W} \sup _{z \in Z} f(z, w) \\
&\Longrightarrow \sup _{z \in Z} \inf _{w \in W} f(z, w) \leq \inf _{w \in W} \sup _{z \in Z} f(z, w)
\end{aligned}
$$

应当注意到的是，上述理论提到的是上界、下界，而非最大值最小值，为了拓展该理论到最大值最小值，需要对函数f(x,y)的形式作出约束：

当约束到特定形式，我们可以得出MiniMax Theorem：

Let $X \subset \mathbb{R}^{n}$ and $Y \subset \mathbb{R}^{m}$ be compact convex sets. If $f: X \times Y \rightarrow \mathbb{R}$ is a continuous function that is concave-convex, i.e. $f(\cdot, y): X \rightarrow \mathbb{R}$ is concave for fixed $y$, and $f(x, \cdot): Y \rightarrow \mathbb{R}$ is convex for fixed $x$
Then we have that
$$
\max _{x \in X} \min _{y \in Y} f(x, y)=\min _{y \in Y} \max _{x \in X} f(x, y)
$$

此时，f对x为凹，对y为凸，我们的等号在鞍点处成立。

当然，这种条件在Lagrange Duality中，将在KKT条件满足时达成。（暂定）

## KKT

