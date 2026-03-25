#---------------------Plot-errors----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle

with open('training_robust.pkl', 'rb') as f:
    training_data = pickle.load(f)

all_results_by_activation = training_data['all_results_by_activation']
best_activation = "exponential"
best_results = all_results_by_activation[best_activation]["best_results"]
best_rule = min(best_results.keys(), key=lambda r: best_results[r]['best_mse'])
x_data = all_results_by_activation[best_activation]["x_data"]
labels = all_results_by_activation[best_activation]["labels"]
t_eval = all_results_by_activation[best_activation]["years"]

def create_gender_separated_plot_pdf(t_eval, x_original, all_results_by_activation, labels,
                                     best_activation='exponential',
                                     best_rule='continuous',
                                     female_pdf="female_results.pdf",
                                     male_pdf="male_results.pdf"):

    sns.set_context("paper", font_scale=3.0)
    sns.set_style("whitegrid")

    gender_styles = {
        "female": {"color": "#cb416b", "linestyle": "-", "shade_color": "lightgrey"},
        "male": {"color": "tab:blue", "linestyle": "-", "shade_color": "lightgrey"}
    }

    female_idx, female_lbl = [], []
    male_idx, male_lbl = [], []

    for i, lbl in enumerate(labels):
        ll = lbl.lower()
        if 'female' in ll:
            female_idx.append(i)
            female_lbl.append(lbl)
        elif 'male' in ll:
            male_idx.append(i)
            male_lbl.append(lbl)

    if female_idx:
        plot_gender_pdf(female_idx, female_lbl, t_eval, x_original, all_results_by_activation,
                        gender_styles["female"], best_activation, best_rule, female_pdf)

    if male_idx:
        plot_gender_pdf(male_idx, male_lbl, t_eval, x_original, all_results_by_activation,
                        gender_styles["male"], best_activation, best_rule, male_pdf)


def plot_gender_pdf(indices, labels, t_eval, x_original, all_results_by_activation,
                    style, best_activation, best_rule, pdf_name):

    n_cols = min(3, len(indices))
    n_rows = (len(indices) + n_cols - 1) // n_cols

    with PdfPages(pdf_name) as pdf:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
        axes = np.atleast_1d(axes).flatten()

        for idx, (v_idx, lbl) in enumerate(zip(indices, labels)):
            plot_single(axes[idx], t_eval, x_original, all_results_by_activation,
                        v_idx, lbl, style, best_activation, best_rule, idx, n_cols)

        for idx in range(len(indices), len(axes)):
            axes[idx].set_visible(False)

        # Ajustar subplots para dejar espacio para la leyenda sin solapamiento
        fig.subplots_adjust(top=0.9, wspace=0.15, hspace=0.5)

        # Crear leyenda global
        handles = [
            plt.Line2D([0], [0], color="black", linestyle=":", linewidth=4, label="Ground truth"),
            plt.Line2D([0], [0], color=style["color"], linestyle="-", linewidth=5, label="cFCM model")
        ]
        fig.legend(handles=handles, loc='upper center', ncol=2,
                   frameon=True, columnspacing=0.5, handletextpad=0.3,
                   handlelength=1.0, bbox_to_anchor=(0.5, 0.99))

        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        plt.close(fig)


def plot_single(ax, t_eval, x_original, all_results_by_activation,
                var_idx, label, style, best_activation, best_rule,
                subplot_idx, n_cols):

    sims_all = []
    for act in all_results_by_activation.keys():
        for rule in all_results_by_activation[act]["best_results"].keys():
            if act == best_activation and rule == best_rule:
                continue
            res = all_results_by_activation[act]["best_results"][rule]
            if 'all_results' in res:
                for r in res['all_results']:
                    if var_idx < r['x_sim'].shape[1]:
                        sims_all.append(r['x_sim'][:, var_idx])
            else:
                x_sim = res['x_sim']
                if var_idx < x_sim.shape[1]:
                    sims_all.append(x_sim[:, var_idx])

    if sims_all:
        sims_arr = np.array(sims_all)
        min_vals = np.min(sims_arr, axis=0)
        max_vals = np.max(sims_arr, axis=0)
        ax.fill_between(t_eval, min_vals, max_vals, alpha=0.5, color=style["shade_color"])

    if best_activation in all_results_by_activation and best_rule in all_results_by_activation[best_activation]["best_results"]:
        x_best = all_results_by_activation[best_activation]["best_results"][best_rule]['x_sim']
        ax.plot(t_eval, x_best[:, var_idx], color=style["color"], linewidth=5, linestyle=style["linestyle"])

    ax.plot(t_eval, x_original[:, var_idx], color="black", linewidth=3, linestyle=":")

    ax.set_xlabel("Year")
    if subplot_idx % n_cols == 0:
        ax.set_ylabel("Score")
    else:
        ax.set_ylabel("")

    label = label.replace('Male', '')
    label = label.replace('Female', '')

    ax.set_title(f"{label}")
    year_range = t_eval[-1] - t_eval[0]
    step = 10 if year_range > 20 else 5
    ax.set_xticks(np.arange(int(t_eval[0]), int(t_eval[-1]) + 1, step))
    ax.set_yticks([0, 0.33, 0.66, 1.0])

    ax.set_yticklabels(['0.0', '0.3', '0.6', '1.0'])
    ax.set_ylim(-0.1, 1.1)


create_gender_separated_plot_pdf(
    t_eval=t_eval,
    x_original=x_data,
    all_results_by_activation=all_results_by_activation,
    labels=labels,
    best_activation='exponential',
    best_rule=best_rule,
    female_pdf="female_models20.pdf",
    male_pdf="male_models20.pdf"
)
#-------------------------------Multimodal Visualizations---------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

index = 2


matrix_df = pd.read_csv(f"medoid_cluster{index}_W_matrix.csv")
matrix = matrix_df.values 


# Compute averages
outgoing_avg = matrix.mean(axis=1)   # row-wise mean
incoming_avg = matrix.mean(axis=0)   # column-wise mean

# Combine averages for coloring
combined_avg = incoming_avg

# Normalize for colormap
norm = plt.Normalize(vmin=combined_avg.min(), vmax=combined_avg.max())
cmap = plt.cm.get_cmap('vlag')  # red = positive, blue = negative
colors = cmap(norm(combined_avg))

sns.set_context("paper", font_scale=2.2)
fig, ax = plt.subplots(figsize=(10,8))

# Swap axes: incoming_avg on x-axis, outgoing_avg on y-axis
scatter = ax.scatter(incoming_avg, outgoing_avg, s=1800, color=colors, edgecolor="black", linewidth=0.0)

for i in range(18):
    ax.text(incoming_avg[i], outgoing_avg[i], f"C{i+1}", ha='center', va='center', color="black")

# Axes lines to show quadrants
ax.axhline(0, color="gray", linestyle="--")
ax.axvline(0, color="gray", linestyle="--")

# Colorbar using the same colormap
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(combined_avg)
fig.colorbar(sm, ax=ax, label="")

ax.set_xlabel("Average Incoming Weight")
ax.set_ylabel("Average Outgoing Weight")

ax.grid(True, linestyle=":")



#-----------------------------------Tsne Visualizations---------------------------------------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

X = flattened

perplexities = [10, 20, 30, 40]
colors = ['#6D071A', '#2C5D8A', '#4B6F44']

sns.set_context("paper", font_scale=3.0)
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

for i, perp in enumerate(perplexities):
    ax = axes[i // 2, i % 2]
    
    tsne2 = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        init='pca',
        learning_rate='auto'
    ).fit_transform(X)
    
    clusters = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    ).fit_predict(tsne2)
    
    for k in np.unique(clusters):
        mask = clusters == k
        points = tsne2[mask]
        
        if len(points) > 2:
            hull = ConvexHull(points)
            hull_vertices = points[hull.vertices]
            ax.fill(
                hull_vertices[:, 0],
                hull_vertices[:, 1],
                color=colors[k],
                alpha=0.1
            )
        
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=300,
            color=colors[k],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.25,
            label=f'Cluster {k+1}'
        )
    
    ax.set_title(f'Perplexity = {perp}')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.grid(True, color='#DADADA', linestyle='-', linewidth=0.6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add outer axes for this subplot
    outer_ax = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes)
    outer_ax.set_xticks([])
    outer_ax.set_yticks([])
    outer_ax.set_frame_on(False)
    outer_ax.set_xlabel("t-SNE 1", labelpad=15)
    outer_ax.set_ylabel("t-SNE 2", labelpad=15)

# Create a single legend above all subplots
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles[:3],
    labels[:3],
    loc='upper center',
    ncol=3,
    frameon=True,
    fancybox=True,
    framealpha=0.5,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig('tsne_perplexity_2x2.pdf', bbox_inches='tight', dpi=300)
plt.show()

plt.savefig(f"concepts_cluster_{index}_methoid.pdf", dpi=800, bbox_inches='tight')
plt.show()
