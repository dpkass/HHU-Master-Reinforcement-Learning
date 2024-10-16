# Exercise set #1

## Exercise 1: Three state MDP

1. $P_0 = \pmatrix{1&0&0}$
2.  see the following table

| $s$ | $a$     | $P$    | $s'$ | $r$  |
| --- | ------- | ------ | ---- | ---- |
| $X$ | $left$  | $1$    | $X$  | $0$  |
| $X$ | $right$ | $0.75$ | $X$  | $1$  |
| $X$ | $right$ | $0.25$ | $Y$  | $-1$ |
| $Y$ | $left$  | $1$    | $X$  | $2$  |
| $Y$ | $right$ | $1$    | $Z$  | $4$  |
| $Z$ | $left$  | $1$    | $Z$  | $0$  |
| $Z$ | $right$ | $1$    | $Z$  | $0$  |
3. calc $R(s,a)$

| $s$ | $a$     | $R$                    |
| --- | ------- | ---------------------- |
| $X$ | $left$  | $0$                    |
| $X$ | $right$ | $0.75*1+0.25*(-1)=0.5$ |
| $Y$ | $left$  | $2$                    |
| $Y$ | $right$ | $4$                    |
| $Z$ | $left$  | $0$                    |
| $Z$ | $right$ | $0$                    |
4. trajectories
	1. $\pi_1 \Rightarrow X,r,-1,Y,r,4,Z$
	2. $\pi_2$ is an endless loop $X,l,0,X,l,\dots$
5. see .ipynb