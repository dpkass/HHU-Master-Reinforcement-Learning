# Exercise set #2

## Exercise 1: Discounted returns

1. calc $G_0,...,G_5$
$$
\displaylines{
G_0 = -1 + \frac{1}{2} * 6 = 2 \\
G_1 = 2 + \frac{1}{2} * 8 = 6 \\
G_2 = 6 + \frac{1}{2} * 4 = 8 \\
G_3 = 3 + \frac{1}{2} * 2 = 4 \\
G_4 = 2 + \frac{1}{2} * 0 = 2 \\
G_5 = 0
}
$$


2. calc $G_0$ for different $\gamma$ with $R_t=5$

$$
\displaylines{
\gamma = 0: G_0 = 5 \\
\gamma = 0.2: G_0 = \sum^{\infty}_{k=0} {0.2^k 5} = 5 \sum^{\infty}_{k=0} {0.2^k} = 5 \frac{1}{1-\frac{1}{5}} = 6.25 \\
\gamma = 0.5: G_0 = 5 * \sum^{\infty}_{k=0} {0.5^k} = 10 \\
\gamma = 0.9: G_0 = 5 * \sum^{\infty}_{k=0} {0.9^k} = 50 \\
\gamma = 0.95: G_0 = 5 * \sum^{\infty}_{k=0} {0.95^k} = 100 \\
\gamma = 0.999: G_0 = 5 * \sum^{\infty}_{k=0} {0.999^k} = 5000 \\
}
$$

3. calc $\gamma$ per $\epsilon,R_T,T$
	1. $0.1 = \gamma^10 * 1 \Rightarrow \gamma = \frac{1}{\root 10 \of 10} \approx 0.7943$
	2.  $0.1 = \gamma^50 * 1 \Rightarrow \gamma = \frac{1}{\root 50 \of 10} \approx 0.9550$
	3. $0.01 = \gamma^50 * 1 \Rightarrow \gamma = \frac{1}{\root 50 \of 100} \approx 0.9120$

## Exercise 2: Value functions
1. $v_\pi(E)=0$, because all following rewards are 0
2. calc $v_{\pi_1}(X)$ and $v_{\pi_1}(Y)$
$$
\displaylines{
v_{\pi}(Y) = \sum_{a} (\pi(a|Y) (R(Y,a) + \gamma \sum_{s'}P(s'|a)v(s'))) \\ 
= \pi(left|Y) (R(Y,a)+\gamma (P(X|Y,left)v_\pi(X)+P(Y|Y,left)v_\pi(Y)+P(Z|Y,left)v_\pi(Z)) + \\
\pi(right|Y) (R(Y,a)+\gamma (P(X|Y,right)v_\pi(X)+P(Y|Y,right)v_\pi(Y)+P(Z|Y,right)v_\pi(Z)) \\
= 0 + 1 (4 + 0.9 (0+0+1*0)) = 4
}
$$

$$
\displaylines{
v_{\pi}(X) = \sum_{a} (\pi(a|X) (R(X,a) + \gamma \sum_{s'}P(s'|a)v(s'))) \\ 
= \pi(left|X) (R(X,a)+\gamma (P(X|X,left)v_\pi(X)+P(Y|X,left)v_\pi(Y)+P(Z|X,left)v_\pi(Z)) + \\
\pi(right|X) (R(X,a)+\gamma (P(X|X,right)v_\pi(X)+P(Y|X,right)v_\pi(Y)+P(Z|X,right)v_\pi(Z)) \\
= 0 + 1 (0.5 + 0.9 (0.75v_\pi(X)+0.25*4+0)) = 1.4 + 0.675v_\pi(X) \\
\\
\implies v_\pi(X)-0.675v_\pi(X) = 1.4 \iff v_\pi(X)=\frac{1.4}{0.325}\approx4.308
}
$$

