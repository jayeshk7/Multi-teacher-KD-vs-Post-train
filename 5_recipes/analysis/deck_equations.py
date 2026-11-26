# loss equations for the slides, rendered with matplotlib mathtext (no latex needed). run: python deck_equations.py <out_dir>
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
BG, INK = "#FAFAF7", "#1F2937"
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 26})

EQ = {
    # pre-checkpoint (1_pre_checkpoint/seq_kd.py): alpha * kl / teacher-answer tokens + (1 - alpha) * ce / ground-truth tokens
    "eq_s1_total": r"$\mathcal{L}(\theta) \;=\; \alpha\,\mathcal{L}_{\mathrm{KD}}(\theta) \;+\; (1-\alpha)\,\mathcal{L}_{\mathrm{CE}}(\theta), \qquad \alpha \in \{1.0,\ 0.5,\ 0.0\}$",
    "eq_s1_kd": r"$\mathcal{L}_{\mathrm{KD}}(\theta) \;=\; \dfrac{1}{N_T}\sum_{i}\;\sum_{t}\;\mathrm{KL}\left(p_T(\cdot \mid x_i, y^T_{i,{<}t})\;\Vert\;p_\theta(\cdot \mid x_i, y^T_{i,{<}t})\right)$",
    "eq_s1_ce": r"$\mathcal{L}_{\mathrm{CE}}(\theta) \;=\; -\dfrac{1}{N_{GT}}\sum_{i}\;\sum_{t}\;\log p_\theta(y^{*}_{i,t} \mid x_i, y^{*}_{i,{<}t})$",
    # gt-sft / t-sft (train_sft_kd.py): nll summed over answer tokens / answer tokens of the 32-row batch
    "eq_sft": r"$\mathcal{L}_{\mathrm{SFT}}(\theta) \;=\; -\dfrac{1}{N}\sum_{i=1}^{B}\;\;\sum_{t=1}^{|y_i|}\;\log p_\theta(y_{i,t} \mid x_i, y_{i,{<}t})$",
    # kd: sum_v p_T (log p_T - log p_S), summed over answer tokens / answer tokens of the batch
    "eq_kd": r"$\mathcal{L}_{\mathrm{KD}}(\theta) \;=\; \dfrac{1}{N}\sum_{i=1}^{B}\;\;\sum_{t=1}^{|y_i|}\;\;\sum_{v \in \mathcal{V}}\;p_T(v \mid x_i, y_{i,{<}t})\,\log\dfrac{p_T(v \mid x_i, y_{i,{<}t})}{p_\theta(v \mid x_i, y_{i,{<}t})}$",
    # grpo advantage, trl scale_rewards="group"
    "eq_adv": r"$\hat{A}_i \;=\; \dfrac{r_i - \mathrm{mean}(r_1, \ldots, r_G)}{\mathrm{std}(r_1, \ldots, r_G) + 10^{-4}}, \qquad G = 8$",
    # trl 1.17.1 loss_type "dapo", beta 0: token mean over all completions of the step
    "eq_grpo": r"$\mathcal{L}_{\mathrm{GRPO}}(\theta) \;=\; -\dfrac{1}{N}\sum_{i=1}^{B}\;\;\sum_{t=1}^{|o_i|}\;\min\left(\rho_{i,t}\,\hat{A}_i,\;\mathrm{clip}(\rho_{i,t},\,1-\epsilon,\,1+\epsilon)\,\hat{A}_i\right)$",
    "eq_rho": r"$\rho_{i,t} \;=\; \dfrac{\pi_\theta(o_{i,t} \mid q_i, o_{i,{<}t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q_i, o_{i,{<}t})}$",
    # one update per batch: rho = 1, so the clip never acts and the gradient is the plain policy gradient
    "eq_grpo_grad": r"$\nabla_\theta \mathcal{L}_{\mathrm{GRPO}} \;=\; -\dfrac{1}{N}\sum_{i=1}^{B}\;\hat{A}_i\;\sum_{t=1}^{|o_i|}\;\nabla_\theta \log \pi_\theta(o_{i,t} \mid q_i, o_{i,{<}t})$",
}


for name, tex in EQ.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, color=INK)
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor=BG)
    plt.close(fig)
    img = plt.imread(f"{OUT}/{name}.png")
    print(name, img.shape[1], "x", img.shape[0], "px -> display", img.shape[1] // 2, "x", img.shape[0] // 2)
