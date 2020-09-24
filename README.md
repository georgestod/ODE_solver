# An experimental ordinary differential equations solver using neural networks
### by Georges Tod, ULB, February 2020
### based on:

Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations." arXiv preprint arXiv:1711.10561 (2017).


This notebook shows how to solve the simple pendulum equation with some friction,

\begin{equation}
\ddot{\theta} + \alpha \cdot \dot{\theta} + \sin{\theta} = 0
\end{equation}

where $\theta$ is the angle of the pendulum and with $\alpha = 0.1$.

Most of the work happens in the attached file: _DE_solver.py_

**REMARK: tensorflow 1.xx is required (not 2.xx)**
