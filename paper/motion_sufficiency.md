# Motion Sufficiency

A chunked policy maps the current observation \(o_t\) and the instruction \(g\) to an action chunk of horizon \(H\),

$$
a_{t:t+H-1}=\pi(o_t,g).
$$

The chunk meets the target only if it is aimed at the target's future positions \(p_{t:t+H}\). Those positions are fixed by which body is selected and how that body moves. The **motion statistic** is the smallest pair that determines them,

$$
\sigma=(\iota,\kappa),\qquad p^{(\iota)}_{t+h}=\Phi_\kappa\!\left(p^{(\iota)}_t,h\right),\quad h=1,\ldots,H.
$$

Here \(\iota\) is the selected body and \(\kappa\) is its motion parameter on a known path family \(\Phi\). The statistic is sufficient when conditioning on it admits an optimal chunk, and no coarser pair does:

$$
\mathrm{MS}(\text{task})=\min\big\{\sigma:\ \exists\,\pi^\star\ \text{s.t.}\ \pi^\star(a_{t:t+H-1}\mid o_t,g,\sigma)\ \text{is optimal}\big\}.
\tag{1}
$$

## Motion

The benchmark uses three families.

$$
\begin{aligned}
\Phi_{v_0}(p,h)&=p+v_0\,h\,\Delta t\,\mathbf{d}, && \kappa=\varnothing, \tag{2}\\
\Phi_{v}(p,h)&=p+v\,h\,\Delta t\,\mathbf{d}, && \kappa=v, \tag{3}\\
\Phi_{\omega}(\theta,h)&=c+R\begin{pmatrix}\cos(\theta+\omega h\Delta t)\\ \sin(\theta+\omega h\Delta t)\end{pmatrix}, && \kappa=\omega. \tag{4}
\end{aligned}
$$

In (2) the straight-belt direction \(\mathbf{d}\) and the speed \(v_0\) are fixed for the whole dataset, so the current position is the statistic. In (3) the speed is drawn once per episode and then held fixed. One RGB frame does not identify it:

$$
o^{\mathrm{rgb}}(p,v)=o^{\mathrm{rgb}}(p,v')\quad\text{for}\quad v\neq v',
$$

while the optimal chunks differ. In (4) the center \(c\) and the radius \(R=0.18\,\mathrm{m}\) are known, and \(\omega\) is the unknown angular speed. The image-plane direction is the tangent at the current angle.

## Selection

Let \(\mathcal{B}_t\) be the bodies on the belt. The instruction binds one of them,

$$
\iota=S(g,o_t)\in\mathcal{B}_t.
$$

When \(|\mathcal{B}_t|=1\), the binding is trivial. Otherwise \(S\) reads the named appearance and, on two belts, the named side \(\beta(\iota)\in\{L,R\}\). The candidates are visible, so \(\iota\) is a function of \((g,o^{\mathrm{rgb}}_t)\) and adds no motion parameter.

## Tasks

Each split has 100 demonstrations. Object count varies inside a split and does not change \(\Phi\).

| Task | \(\iota=S(g,o_t)\) | \(\kappa\) | \(\sigma\) |
|---|---|---|---|
| block | \(\|\mathcal{B}_t\|=1\) | \(\varnothing\) in (2) | \(p_t\) |
| rand_speed | \(\|\mathcal{B}_t\|=1\) | \(v\) in (3) | \((p_t,v)\) |
| multi | named body, one belt | \(v\) in (3) | \((p^{(\iota)}_t,v)\) |
| dual | named body and belt | \(v\) in (3) | \((p^{(\iota)}_t,v)\) |
| ring | named body | \(\omega\) in (4) | \((\theta^{(\iota)}_t,\omega)\) |

On multi and dual, \(|\mathcal{B}_t|\in\{2,3,4\}\) and \(v\in[0.10,0.59]\,\mathrm{m/s}\). On ring, the radius is \(0.18\,\mathrm{m}\) and the grasp angle lies in \([-0.35,0.35]\,\mathrm{rad}\), with the same object count and a speed in \([0.10,0.52]\,\mathrm{m/s}\).

## Event readout

Let \(E_t\) be the event volume on \((t-\tau,t]\) with \(\tau=0.4\,\mathrm{s}\). The motion head predicts four displacements of the target center,

$$
\hat\Delta_h=\psi(E_t),\qquad h\in\{4,8,12,16\},\qquad \Delta_h=p_{t+h}-p_t.
$$

They are a coordinate form of \(\kappa\), tied to the two path families by

$$
\begin{aligned}
\det[\Delta_{h_1},\Delta_{h_2}]&=0, && \|\Delta_h\|\propto h && \text{for (3)}, \tag{5}\\
\|p_{t+h}-c\|&=R, && \det[\Delta_{h_1},\Delta_{h_2}]\neq 0 && \text{for (4)}. \tag{6}
\end{aligned}
$$

The selected center \(q_t=p^{(\iota)}_t\) is added to the event tokens that the action bridge reads, so the bridge is conditioned on \(\iota\). The pool inside \(\psi\) is taken before this addition. It recovers \(\kappa\) of the target when \(|\mathcal{B}_t|=1\). When \(|\mathcal{B}_t|>1\), every body contributes to the same pool, and \(\iota\) is carried by \(q_t\) rather than by \(\psi\).

## Checks

Write \(\pi_E\) for the policy that sees events and \(\pi_R\) for the RGB policy. Sufficiency gives four equalities.

$$
\begin{aligned}
\pi_R(o^{\mathrm{rgb}},g)&\approx\pi_E && \text{on block}, \tag{7}\\
\pi_R(o^{\mathrm{rgb}},g,v)&\approx\pi_E && \text{on (3)}, \tag{8}\\
\pi_R(o^{\mathrm{rgb}},g,q_t,v)&\approx\pi_E && \text{on multi and dual}, \tag{9}\\
\pi_R(o^{\mathrm{rgb}},g,\theta_t,\omega)&\approx\pi_E && \text{on ring}. \tag{10}
\end{aligned}
$$

In (9) the two oracles separate the factors: \(q_t\) removes the error in \(\iota\), and \(v\) removes the error in \(\kappa\). On dual, the belt bit \(\beta(\iota)\) drops out once \(q_t\) is given. Equation (5) is the geometric test on the straight belts, and (6) is the test on the ring.
