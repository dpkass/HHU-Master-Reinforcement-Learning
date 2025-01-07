Given:
$$
\pi_\omega(a|s) = \frac{\exp\left(x(s,a)^\top\omega\right)}{\sum_{a'}\exp\left(x(s,a)^\top\omega\right)}
$$

Show:
$$
\nabla_{\omega} \log \pi_\omega(a|s) = x(s,a)-\sum_{a'} \pi_{\omega}(a'|s)x(s,a')
$$

$$
\begin{align}
\nabla_{\omega} \log \pi_\omega(a|s)
&= \nabla_{\omega} \log \frac{\exp\left(x(s,a)^\top\omega\right)}{\sum_{a'}\exp\left(x(s,a')^\top\omega\right)} \\
&= \nabla_{\omega} \left[ \log \exp\left(x(s,a)^\top\omega\right) - \log \sum_{a'}\exp\left(x(s,a')^\top\omega\right) \right] \\
&= \nabla_{\omega} x(s,a)^\top\omega - \nabla_{\omega} \log \sum_{a'}\exp\left(x(s,a')^\top\omega\right) \\
&= x(s,a) - \frac{\nabla_{\omega} \sum_{a'}\exp\left(x(s,a')^\top\omega\right)}{\sum_{a'}\exp\left(x(s,a')^\top\omega\right)} \\
&= x(s,a) - \frac{\sum_{a'}x(s,a')\exp\left(x(s,a')^\top\omega\right)}{\sum_{a'}\exp\left(x(s,a')^\top\omega\right)} \\
&= x(s,a) - \sum_{a'}\frac{\exp\left(x(s,a')^\top\omega\right)}{\sum_{a''}\exp\left(x(s,a'')^\top\omega\right)}x(s,a') \\
&= x(s,a)-\sum_{a'} \pi_{\omega}(a'|s)x(s,a')
\end{align}
$$
