# DynaBench

DynaBench is a conveyor-grasping suite for comparing vision-language-action policies under RGB and event sensing. A right-arm UR3e picks up the object named in the instruction and places it in the named tray. Five splits share this goal, the same arm, and the same household object set. Each split has 100 demonstrations. They differ only in the motion statistic below, which is the quantity a chunked policy has to recover before it commits to the grasp.

## Setup

At time \(t\) the benchmark exposes an instruction \(g\) and an observation

$$
o_t=\big(I^{\mathrm{head}}_t,\ I^{\mathrm{wrist}}_t,\ E^{\mathrm{head}}_t,\ E^{\mathrm{wrist}}_t,\ x_t\big).
$$

\(I^{\mathrm{head}}_t\) and \(I^{\mathrm{wrist}}_t\) are RGB frames from the head camera and the right wrist. \(E^{\mathrm{head}}_t\) and \(E^{\mathrm{wrist}}_t\) are the corresponding event streams, each a list of \((x,y,t,p)\). \(x_t\in\mathbb{R}^7\) is the right-arm proprioception, six joint positions and the gripper. The action is an absolute command in the same seven-dimensional space. A policy may emit one step or a chunk of horizon \(H\),

$$
a_{t:t+H-1}=\pi(o_t,g).
$$

An episode succeeds when the named object is in the named tray. Frame annotations provide the target center \(q_t\) and its later positions, so the displacements below can be computed from the release without a separate labeling pass.

## Event simulation

The event streams are not differences of the saved RGB frames. Inside the physics loop, at \(\Delta t_{\mathrm{phys}}=1/250\,\mathrm{s}\), each event camera renders its own color image and converts it to linear luminance,

$$
L=0.2126\,R+0.7152\,G+0.0722\,B.
\tag{11}
$$

A pixel stores the log luminance, with \(\varepsilon=10^{-3}\),

$$
\ell=\log\big(\max(L,0)+\varepsilon\big),
$$

and a reference value \(\ell^\star\). At a sample time \(t\),

$$
\delta=\ell(t)-\ell^\star.
$$

The head camera and the right wrist use the same thresholds, \(C_+=C_-=0.2\). A pixel emits events only when the log change crosses a threshold, and at most \(N_{\max}=10\) events per sample:

$$
\begin{aligned}
n_+&=\min\Big(N_{\max},\ \big\lfloor \delta/C_+\big\rfloor\Big) && \text{if }\delta\ge C_+,\\
n_-&=\min\Big(N_{\max},\ \big\lfloor -\delta/C_-\big\rfloor\Big) && \text{if }\delta\le -C_-,
\end{aligned}
\tag{12}
$$

and \(n_+=n_-=0\) otherwise. The reference then advances by the thresholds that were crossed, so the residual change is kept,

$$
\ell^\star\leftarrow \ell^\star+n_+C_+-n_-C_-.
\tag{13}
$$

The refractory period is zero, and both cameras are sampled on every physics step. For a pixel that fires \(n=n_++n_-\) times, the events are \((x,y,t_k,p)\), with polarity \(p=+1\) when \(n_+>0\) and \(p=-1\) when \(n_->0\). Their timestamps divide the physics step evenly,

$$
t_k=t-\Delta t_{\mathrm{phys}}+\frac{k}{n}\,\Delta t_{\mathrm{phys}},\qquad k=1,\ldots,n.
$$

A positive event is an increase of log luminance and a negative event is a decrease. A faster object crosses \(C_+\) or \(C_-\) more often, so the number of events in a fixed window grows with speed. That is the sensing fact behind \(\kappa\) in (3) and (4): the saved RGB frame can share one appearance across many speeds, while the event count in the same interval does not.

## Motion statistic

The chunk succeeds only if it is aimed at where the target will be. That future is fixed by which body is selected and how the selected body moves. The motion statistic is the smallest pair that determines the future positions \(p_{t:t+H}\),

$$
\sigma=(\iota,\kappa),\qquad p^{(\iota)}_{t+h}=\Phi_\kappa\!\left(p^{(\iota)}_t,h\right),\quad h=1,\ldots,H.
$$

\(\iota\) is the selected body and \(\kappa\) is its parameter on a known path family \(\Phi\). The statistic of a task is the minimal sufficient pair:

$$
\mathrm{MS}(\mathrm{task})=\min\big\{\sigma:\ \exists\,\pi^\star\ \text{s.t.}\ \pi^\star(a_{t:t+H-1}\mid o_t,g,\sigma)\ \text{is optimal}\big\}.
\tag{1}
$$

Three path families occur. Each one is numbered in the text so the table can point back to it.

**(2) Fixed straight belt.** The direction \(\mathbf{d}\) and the speed \(v_0\) are the same in every episode, so \(\kappa\) is empty and the current position fixes the future:

$$
\Phi_{v_0}(p,h)=p+v_0\,h\,\Delta t\,\mathbf{d}.
$$

**(3) Straight belt, speed drawn once per episode.** The direction stays \(\mathbf{d}\), and the episode speed \(v\) is constant after it is drawn. Then \(\kappa=v\):

$$
\Phi_{v}(p,h)=p+v\,h\,\Delta t\,\mathbf{d}.
$$

**(4) Ring.** The center \(c\) and the radius \(R=0.18\,\mathrm{m}\) are known. The unknown is the angular speed \(\omega\), so \(\kappa=\omega\), and the image-plane direction is the tangent at the current angle:

$$
\Phi_{\omega}(\theta,h)=c+R\begin{pmatrix}\cos(\theta+\omega h\Delta t)\\ \sin(\theta+\omega h\Delta t)\end{pmatrix}.
$$

Under (2) the current position is enough. Under (3) a single RGB frame does not identify the speed,

$$
o^{\mathrm{rgb}}(p,v)=o^{\mathrm{rgb}}(p,v')\quad\text{for}\quad v\neq v',
$$

and the optimal chunks for \(v\) and \(v'\) differ. Under (4) the lead has to follow that tangent, so the future centers trace an arc of radius \(R\).

Let \(\mathcal{B}_t\) be the bodies on the belts. Selection reads the instruction and the current image,

$$
\iota=S(g,o^{\mathrm{rgb}}_t)\in\mathcal{B}_t.
$$

The binding is trivial when \(|\mathcal{B}_t|=1\). Otherwise \(S\) returns the named appearance and, when two belts are present, the named side \(\beta(\iota)\in\{L,R\}\). Every candidate is visible, so \(\iota\) adds no entry to \(\kappa\).

Four annotated displacements are the coordinate form of \(\kappa\),

$$
\Delta_h=p_{t+h}-p_t,\qquad h\in\{4,8,12,16\}.
$$

They lie on the path that generated them:

$$
\begin{aligned}
\det[\Delta_{h_1},\Delta_{h_2}]&=0, && \|\Delta_h\|\propto h && \text{under (3)}, \tag{5}\\
\|p_{t+h}-c\|&=R, && \det[\Delta_{h_1},\Delta_{h_2}]\neq 0 && \text{under (4)}. \tag{6}
\end{aligned}
$$

## Tasks

The five tasks are built from one grasp-and-place goal by changing the belt layout, the speed, and how the instruction names the target. Table 1 lists the resulting statistic.

| Task | What is placed on the belt | Motion formula | \(\sigma\) |
|---|---|---|---|
| block | one object, straight belt, speed fixed at \(v_0\) | (2) | \(p_t\) |
| rand_speed | one object, straight belt, a new speed \(v\) each episode | (3) | \((p_t,v)\) |
| multi | \(2\)–\(4\) objects on one straight belt, \(v\in[0.11,0.54]\,\mathrm{m/s}\) | (3) | \((p^{(\iota)}_t,v)\) |
| dual | \(2\)–\(4\) objects on a left or a right belt, \(v\in[0.10,0.59]\,\mathrm{m/s}\) | (3) | \((p^{(\iota)}_t,v)\) |
| ring | \(2\)–\(4\) objects on a ring of radius \(0.18\,\mathrm{m}\) | (4) | \((\theta^{(\iota)}_t,\omega)\) |

**block.** One household object rides a straight belt at a single speed \(v_0\) shared by every episode. The instruction names that object and the destination tray, for example “grasp the bell and put it in the blue tray.” Its future position is equation (2), so the grasp is fixed by where the object is now.

**rand_speed.** The scene is unchanged: one object, one straight belt, one tray. The belt speed is drawn separately for each episode and then held constant, so the future position is equation (3). Two episodes can show the object at the same image location and still require different leads.

**multi.** Each episode places two, three, or four objects in a row on one straight belt, three on average, with speed drawn from \([0.11,0.54]\,\mathrm{m/s}\). The instruction names one of them, for example “pick up the apple.” The chosen object still moves by equation (3). The added demand is \(\iota=S(g,o^{\mathrm{rgb}}_t)\): the other objects are visible distractors on the same belt.

**dual.** Two straight belts run side by side, one on the left and one on the right. An episode places two to four objects and draws the speed from \([0.10,0.59]\,\mathrm{m/s}\). The instruction names both the object and the belt, for example “grasp the mouse on the left conveyor.” The chosen object still moves by equation (3). Selection now returns the appearance together with the side, \(\beta(\iota)\in\{L,R\}\).

**ring.** The belt is a circle of radius \(R=0.18\,\mathrm{m}\). Two to four objects travel on it, and the target angle at the grasp lies in \([-0.35,0.35]\,\mathrm{rad}\). The instruction names the object, as in “lift the mouse off the loop.” Selection matches `multi`. The future position is equation (4): the lead follows the tangent at \(\theta^{(\iota)}_t\), and the future centers trace an arc of radius \(R\). The belt speed along the ring lies in \([0.10,0.52]\,\mathrm{m/s}\).

## Protocol

Policies are trained on the released demonstrations and scored by success on held-out episodes of the same split. The observation above supports an RGB policy \(\pi_R\), which drops both event streams, and an event policy \(\pi_E\), which keeps them. Motion sufficiency predicts when the two agree. Writing the extra arguments as oracles logged with the episode,

$$
\begin{aligned}
\pi_R(o^{\mathrm{rgb}},g)&\approx\pi_E && \text{on block}, \tag{7}\\
\pi_R(o^{\mathrm{rgb}},g,v)&\approx\pi_E && \text{on equation (3)}, \tag{8}\\
\pi_R(o^{\mathrm{rgb}},g,q_t,v)&\approx\pi_E && \text{on multi and dual}, \tag{9}\\
\pi_R(o^{\mathrm{rgb}},g,\theta_t,\omega)&\approx\pi_E && \text{on ring}. \tag{10}
\end{aligned}
$$

Equation (7) is the fixed-speed control. Equation (8) attributes a `rand_speed` gap to \(v\): the RGB policy given the true speed should match the event policy. In (9) the oracles separate the two factors on `multi` and `dual`. Supplying \(q_t\) removes the error in \(\iota\); supplying \(v\) removes the error in \(\kappa\). Once \(q_t\) is given, the belt bit \(\beta(\iota)\) on `dual` adds nothing. Equation (10) is the same check for the ring, with \((\theta_t,\omega)\) in place of \((p_t,v)\).

For a method that predicts \(\Delta_h\), (5) and (6) are geometric checks on the same episodes. The straight-belt splits should satisfy (5). The ring should satisfy (6).
