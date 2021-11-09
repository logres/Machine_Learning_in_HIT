# EM算法

EM算法并非一种机器学习模型，而是一种对于“双未知”“先鸡先蛋”问题的一种学习/优化方案。我们可以在许多算法模型中见到其身影，K-means聚类、GMM模型、维比特算法等等。Today, I want to show the true essence of EM algorism here and explain it from a perspective of convex optimization.

## What we need EM to do ?

Although I think all the people read this note here already have a clear knowledge of or at least have known some of the algorism refering EM, I still need to give the explanation of what the problems EM could do for us.

以GMM举例，我们需要为一组数据建模，然而他们在特征空间的分布可能无法使用传统的简单模型进行分析，一组不同的高斯分布也许能为我们解决这个问题，也即所谓的高斯混合模型。

当我们尝试拟合一个高斯模型时（调整$\mu \ \Sigma$使得高斯模型在数据上达到最大似然/最大后验），一切是自然而简单的。我们知道每个数据样本属于这个高斯，用来拟合他就完事了。

当视角切换至混合高斯，我们只知道这个样本属于其中一个高斯，但不知道是哪一个。也许我们可以从pdf最大化的角度来判断其属于哪一个高斯，但问题在于，我们手中的高斯并未拟合到其应有的样子。这么的，需要拟合高斯，我们需要判断样本点的所属；需要判断样本点的所属，我们需要拟合好的高斯——一切陷入僵局。

EM algorism 正是破局之法。

## How does EM do ?

EM consists of two part: E-Exception M-Maximization

### E-Exception

In E step, we suppose we have get the knowledge of the distribution of Gaussion/centroids/trans-prob, whatever we need to know for our training,of the Model.

And we start to assign our sample to the most possible one, and believe that is the type of the sample, although we know that's just a suppose, not truth.

### M-Maximization

In M step, we need to paid for our supposes. Now, since we have get the type of all the samples, we can start learning/training.

After training, we get a set of new distribution of more likelihood with samples assigned to it.

If the assignment we give to samples(zero-one labels or weight to every distribution) is true, now we have done all the work we need to do.

Unfortunately, that is just a suppose. Since we give a incorrect assignment, the result we get now is not sure to be correct. But, it is somehow more close to the true one, and we can once again suppose the distribution is true, and start next E step.

### total

After the discussion of M and E above, now we can see the whole appearance of EM algorism.

Repeat until convergence \{
(E-step) For each $i$, set
$$
\begin{aligned}
&Q_{i}\left(z^{(i)}\right):=p\left(z^{(i)} \mid x^{(i)} ; \theta\right) \\
(\text { M-step) Set } \\
&\theta:=\arg \max _{\theta} \sum_{i} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)}
\end{aligned}
$$ \}

In E-step, we assign all samples marked as i according to the distribution we have now marked as $\theta$, getting the assignment Q. And in M-step, we compute the distribution $\theta$ according to the assignment Q we make in E-step.
We will do these until convergence when Q and $\theta$ will not change anymore.

## Why does EM work?

In some algorism, such as K-means, "EM will working well" seems is an intuition for us. But that intuition seems not so true in more complex algorism such as GMM. So it seems we need some math foundmentation for EM.

### Jensen Inequality

To get some tool for the proof of our EM algorism, now we need to introduce Jensen Inequality first.

**Theorem.** Let $f$ be a convex function, and let $X$ be a random variable. Then:
$$
\mathrm{E}[f(X)] \geq f(\mathrm{E} X)
$$
Moreover, if $f$ is strictly convex, then $\mathrm{E}[f(X)]=f(\mathrm{E} X)$ holds true if and only if $X=\mathrm{E}[X]$ with probability 1 (i.e., if $X$ is a constant).

It holds true for concave function, but with contrast symbol.

### proof of EM

We start our proof of set the goal.

Setting our likelihood as $\ell(\theta)$ a function of $\theta$. When including unknown distribution z, it can be expressed in the way below.

$$\begin{aligned}
\ell(\theta) &=\sum_{i=1}^{m} \log p(x ; \theta) \\
&=\sum_{i=1}^{m} \log \sum_{z} p(x, z ; \theta)
\end{aligned}$$

now, start M-step. We give samples a distribution Q.

$$
\begin{aligned}
\sum_{i} \log p\left(x^{(i)} ; \theta\right) &=\sum_{i} \log \sum_{z^{(i)}} p\left(x^{(i)}, z^{(i)} ; \theta\right) \\
&=\sum_{i} \log {\sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)}} \\
& \geq \sum_{i} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)}
\end{aligned}
$$

We can naturally get the second line and Jensen Inequality will help to get the third line, for:

$$
f\left(\mathrm{E}_{z^{(i)} \sim Q_{i}}\left[\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)}\right]\right) \geq \mathrm{E}_{z^{(i)} \sim Q_{i}}\left[f\left(\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)}\right)\right]
$$

当X为常量时，等号成立。此时：

$$\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)} = c$$

也即 $$Q_{i}\left(z^{(i)}\right) = c \ p\left(x^{(i)}, z^{(i)} ; \theta\right)$$

且$$\sum_z Q_{i}\left(z^{(i)}\right) = 1$$

故
$$
\begin{aligned}
Q_{i}\left(z^{(i)}\right) &=\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{\sum_{z} p\left(x^{(i)}, z ; \theta\right)} \\
&=\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{p\left(x^{(i)} ; \theta\right)} \\
&=p\left(z^{(i)} \mid x^{(i)} ; \theta\right)
\end{aligned}
$$

于是便得到了EM算法：

Repeat until convergence \{
(E-step) For each $i$, set
$$
\begin{aligned}
&Q_{i}\left(z^{(i)}\right):=p\left(z^{(i)} \mid x^{(i)} ; \theta\right) \\
(\text { M-step) Set } \\
&\theta:=\arg \max _{\theta} \sum_{i} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)}
\end{aligned}
$$ \}

这个过程十分有趣，在E我们设置Q，构造$\ell(\theta)$的下确界，而后在M，我们计算$\theta$，使得这个下确界取得最大值。然后重复这个过程，直至收敛。我们设下界为$h_t(\theta)$,有：

$$\ell(\theta^{t+1})\geq h_{t}(\theta^{t+1}) \geq h_t(\theta^t)=\ell(\theta^{t})$$

In this iteration, our $\ell(\theta^{t})$ will increase as t increase, and somehow stop in some optimal(global or local).

@import "/EM.png"

**That's all for EM algorism.**
