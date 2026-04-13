import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx




#----------------------------------------------------------------

def load_depression_data(file_path, activation_func, column='val'):
   
    dt = pd.read_csv(file_path)
    dalys = dt[dt['measure_name'] == 'DALYs (Disability-Adjusted Life Years)']
    mtr = dalys.pivot_table(index='year', columns=['age_name','sex_name'], values=column)

    x = mtr.values
    years = mtr.index.values
    labels = [f"{a} {s}" for (a, s) in mtr.columns]

    eps = 1e-8
    xmin = x.min(axis=0, keepdims=True)
    xmax = x.max(axis=0, keepdims=True)

    if activation_func == 'exponential' or activation_func =='sigmoid':
        x_sc = (x - xmin) / (xmax - xmin + eps)
    elif activation_func == 'tanh':
        x_sc = 2 * (x - xmin) / (xmax - xmin + eps) - 1
    else:
        raise ValueError("Invalid activation function")

    return x_sc, years, labels
#----------------------------------------------------------------

def _exponential(X, beta, U=10):
    return np.sign(X) * np.minimum(np.abs(X), U ** (1.0 / beta)) ** beta

def activation(x, activation_func, beta=1.0):
    if activation_func == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation_func == 'tanh':
        return np.tanh(x)
    elif activation_func == 'exponential':
        return _exponential(x, beta)
    else:
        raise ValueError("Invalid activation function")
#--------------------------------------------------------------------
def simulate_fcm(x0, W, b, t_eval, alpha=1.0, beta=2.5, phi=0.8,
                 rule='continuous', activation_func='exponential'):
    steps = len(t_eval)
    x = np.zeros((steps, len(x0)))
    x[0] = x0

    for i in range(1, steps):
        if rule == 'type-I':
            x[i] = activation(np.dot(x[i-1], W.T), activation_func)
        elif rule == 'type-II':
            x[i] = activation(x[i-1] + np.dot(x[i-1], W.T), activation_func)
        elif rule == 'quasi-nonlinear':
            x[i] = activation((1-phi) * x[0] + phi * np.dot(x[i-1], W.T), activation_func)
        elif rule == 'continuous':
            x[i] = alpha * (x[i-1] - activation(np.dot(x[i-1], W.T) + b, activation_func, beta))
        else:
            raise ValueError("Invalid reasoning rule.")
    return x
#----------------------------------------------------------------------------------------------------
def fcm_residuals(flat_params, t_data, x_data, lambda_w=1e-8, lambda_b=1e-4, 
                  rule='continuous', activation_func='exponential', alpha=1.0, beta=2.5, phi=0.8):
    n = x_data.shape[1]
    K = x_data.shape[0]
    M = K * n
    W = flat_params[:n*n].reshape((n, n))
    b = flat_params[n*n:]
    x0 = x_data[0]

    x_sim = simulate_fcm(x0, W, b, t_data, alpha=alpha, beta=beta, phi=phi, 
                         rule=rule, activation_func=activation_func)

    err = x_sim - x_data
    res_data = err.ravel() * np.sqrt(2 / M)
    res_w = W.ravel() * np.sqrt(2 * lambda_w)
    res_b = b * np.sqrt(2 * lambda_b)
    return np.concatenate((res_data, res_w, res_b))


def train_fcm(x, lambda_w=1e-8, lambda_b=1e-4, n_repetitions=1,
              rule='continuous', activation_func='exponential', alpha=1.0, beta=2.5, phi=0.8,
              verbose=False):
    t_data = np.arange(x.shape[0])
    n = x.shape[1]

    best_loss = np.inf
    best_W = None
    best_b = None

    lower = -2 * np.ones(n*n + n)
    upper = 2 * np.ones(n*n + n)
    bounds = (lower, upper)

    for rep in range(n_repetitions):
        flat_init = np.random.uniform(-0.1, 0.1, size=n*n + n)
        res = least_squares(
            fcm_residuals, flat_init,
            args=(t_data, x, lambda_w, lambda_b, rule, activation_func, alpha, beta, phi),
            method='trf', bounds=bounds, max_nfev=2000
        )
        final_loss = res.cost
        if final_loss < best_loss:
            best_loss = final_loss
            best_W = res.x[:n*n].reshape((n, n))
            best_b = res.x[n*n:]
        
    return best_W, best_b, best_loss

def train_once(x_data, rule, activation_func, param_value, verbose=False):
    kwargs = dict(rule=rule, activation_func=activation_func)
    if rule == 'quasi-nonlinear':
        kwargs['phi'] = param_value
    if rule == 'continuous':
        kwargs['beta'] = param_value

    W, b, loss = train_fcm(x_data, n_repetitions=5, verbose=verbose, **kwargs)

    x0 = x_data[0]
    t_eval = np.arange(len(x_data))
    x_sim = simulate_fcm(x0, W, b, t_eval, **kwargs)

    mse = np.mean((x_data - x_sim)**2)
    rmse = np.sqrt(mse)
    
    mae = np.mean(np.abs(x_data - x_sim))

    return {
        'parameter': param_value,
        'W': W,
        'b': b,
        'loss': loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'x_sim': x_sim
    }

def find_best_parameter(x_data, activation_func, verbose=False):
 
    reasoning_rules = ['continuous', 'type-I', 'type-II', 'quasi-nonlinear']
    param_quas = [0.2, 0.5, 0.8]
    param_cont = [1.0, 1.5, 2.0, 2.5]

    best_results = {}

    for rule in reasoning_rules:
        if rule == 'quasi-nonlinear':
            params = param_quas
        elif rule == 'continuous':
            params = param_cont
        else:
            params = [None]

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(train_once)(x_data, rule, activation_func, pv, verbose=False) for pv in params
        )

        best_result = min(results, key=lambda x: x['mse'])
        best_results[rule] = {
            'best_parameter': best_result['parameter'],
            'best_mse': best_result['mse'],
            'best_rmse': best_result['rmse'],
            'best_mae': best_result['mae'],
            'best_loss': best_result['loss'],
            'W': best_result['W'],
            'b': best_result['b'],
            'x_sim': best_result['x_sim'],
            'all_results': results
        }

        best_param_txt = (
            f"beta={best_result['parameter']}" if rule=='continuous' else
            f"phi={best_result['parameter']}" if rule=='quasi-nonlinear' else "N/A"
        )
       
        print(f"[{activation_func}] {rule:>15}, best MSE={best_result['mse']:.3e} ({best_param_txt})")

    return best_results
 


def summarize_best_results(best_results, activation):
    
    rows = []
    for rule in ['continuous', 'type-I', 'type-II', 'quasi-nonlinear']:
        if rule not in best_results:
            continue
        br = best_results[rule]
        if rule == 'continuous':
            best_param = f"beta={br['best_parameter']}"
        elif rule == 'quasi-nonlinear':
            best_param = f"phi={br['best_parameter']}"
        else:
            best_param = "N/A"

        rows.append({
            'Activation_Function': activation,
            'Reasoning_Rule': rule,
            'Best_Parameter': best_param,
            'MSE': br['best_mse'],
            'RMSE': br['best_rmse'],
            'MAE': br['best_mae'],
            'Loss': br['best_loss']
        })
    df = pd.DataFrame(rows)
    return df


def sensitivity_analysis(file_path, best_W, best_b, activation_func='exponential', rule='continuous', beta=2.5):
    scenarios = ['val', 'lower', 'upper']
    polarity = {}
    importance = {}
    
    base_W = best_W.copy()

    for sce in scenarios:
        x_sce, _, _ = load_depression_data(file_path, activation_func,sce)
        print(f"{sce}: min={x_sce.min():.6f}, max={x_sce.max():.6f}, mean={x_sce.mean():.6f}")
        
        node_importance = np.sum(np.abs(base_W), axis=1) * x_sce
        importance[sce] = node_importance
        
        node_polarity = np.sign(np.mean(base_W, axis=1))
        polarity[sce] = node_polarity
        
        print(f"{sce.upper()} done")
    
                

    corr_lower, p_lower = spearmanr(importance['val'], importance['lower'])
    corr_upper, p_upper = spearmanr(importance['val'], importance['upper'])
    print(f"\n  Mean importance - val: {np.mean(importance['val']):.6f}")
    print(f"  Mean importance - lower: {np.mean(importance['lower']):.6f}")
    print(f"  Mean importance - upper: {np.mean(importance['upper']):.6f}")
    
    stable = np.all(polarity['val'] == polarity['lower']) and np.all(polarity['val'] == polarity['upper'])
    
  
    print(f"val vs lower:{corr_lower.flatten()[0]:.3f} (p={p_lower.flatten()[0]:.4f})")
    print(f"val vs upper:{corr_upper.flatten()[0]:.3f} (p={p_upper.flatten()[0]:.4f})")
    print(f"Polarity stable: {stable}")

    return importance, polarity






file_path = "data_depression.csv"
activations_to_run = ["exponential", "tanh", "sigmoid"]  

all_tables = []
all_results_by_activation = {}  
for act in activations_to_run:
    x_data, years, labels = load_depression_data(file_path, act)
    best_results = find_best_parameter(x_data, activation_func=act, verbose=False)
    all_results_by_activation[act] = {
        "x_data": x_data, "years": years, "labels": labels, "best_results": best_results
    }
    df_act = summarize_best_results(best_results, act)
    all_tables.append(df_act)

df_all = pd.concat(all_tables, ignore_index=True)


display_cols = ["Activation_Function", "Reasoning_Rule", "Best_Parameter", "MSE", "RMSE", "MAE", "Loss"]
print(df_all[display_cols].sort_values(["Activation_Function","MSE"]).to_string(index=False))


df_all_sorted = df_all.sort_values(["Activation_Function","MSE"]) 
df_all_sorted.to_csv("best_results_summary_all.csv", index=False)




best_results_exp = all_results_by_activation['exponential']['best_results']
best_continuous = best_results_exp['continuous']
best_W = best_continuous['W']
best_b = best_continuous['b']
best_beta = best_continuous['best_parameter']



sensitivity_analysis(file_path, best_W, best_b, 
                    activation_func='exponential', rule='continuous', beta=best_beta)






#---------------------Multimodal analysis-------------------------------

best_activation = "exponential" 
best_results = all_results_by_activation[best_activation]["best_results"]
best_rule = min(best_results.keys(), key=lambda r: best_results[r]['best_mse'])
best_param = best_results[best_rule]['best_parameter']

x_data = all_results_by_activation[best_activation]["x_data"]
labels = all_results_by_activation[best_activation]["labels"]

n_retrainings = 20 
n_seeds = 5  
low = 1e-4
high = 1e-3
max_ = 500

lam_w = best_results[best_rule].get("best_lambda", 1e-08)
lam_b = best_results[best_rule].get("best_lambda_b", lam_w)


all_W, all_mse = [], []

for retrain_idx in range(n_retrainings):
    print(f"\nRetraining {retrain_idx+1}/{n_retrainings}")
    for seed in range(n_seeds):
        current_seed = retrain_idx * n_seeds + seed
        best_mse = np.inf
        best_W = None
        tries_used = 0

        for t in range(max_):
            np.random.seed(current_seed + t)

            # train
            W, b, _ = train_fcm(
                x_data,
                n_repetitions=5,
                verbose=False,
                lambda_w=lam_w,
                lambda_b=lam_b,
                rule="continuous",                 
                activation_func=best_activation,
                beta=best_param                 
            )

            
            x0 = x_data[0]
            t_eval = np.arange(len(x_data))
            x_sim = simulate_fcm(
                x0, W, b, t_eval,
                rule="continuous",
                activation_func=best_activation,
                beta=best_param
            )
            mse = np.mean((x_data - x_sim)**2)

            
            if mse < best_mse:
                best_mse, best_W = mse, W

            tries_used = t + 1
            if low <= best_mse <high: 
               print(best_mse)
               break
            else:
                print("i didnt find optimal mse")

        all_W.append(best_W)
        all_mse.append(best_mse)
        print(f"  seed {seed+1}/{n_seeds}: best MSE={best_mse:.3e} (tries={tries_used})")

all_W = np.array(all_W)
print(f"\nCollected {all_W.shape[0]} matrices")
print(f"MSE range: [{np.min(all_mse):.3e}, {np.max(all_mse):.3e}]")
print(f"Mean MSE: {np.mean(all_mse):.3e} ± {np.std(all_mse):.3e}")

# flatten
n_mats, nrows, ncols = all_W.shape
flattened = all_W.reshape(n_mats, nrows * ncols)





with open('weight_matrix.pkl', 'rb') as f:
    weight_matrix = pickle.load(f)  

flattened = weight_matrix['all_weight_matrix']
all_w = weight_matrix['w']

mse_los = weight_matrix['mse']


#-----------------------------------------------------------------------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(flattened)

medoid_indices = []
medoid_mses = []
cluster_sizes = []

for cluster_id in range(3):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_sizes.append(len(cluster_indices))
    
    cluster_data = flattened[cluster_indices]
    distances = euclidean_distances(cluster_data)
    
    medoid_idx_in_cluster = np.argmin(distances.sum(axis=1))
    medoid_global_idx = cluster_indices[medoid_idx_in_cluster]
    
    medoid_indices.append(medoid_global_idx)
    medoid_mses.append(mse_los[medoid_global_idx])
    
    print(f"\nCluster {cluster_id}: {len(cluster_indices)} runs")
    print(f"  Medoid index: {medoid_global_idx}, MSE: {mse_los[medoid_global_idx]:.6f}")

medoids_df = pd.DataFrame({
    'Cluster': [0, 1, 2],
    'Global_Index': medoid_indices,
    'MSE': medoid_mses,
    'Cluster_Size': cluster_sizes
})

medoids_df.to_csv('medoids_results.csv', index=False)



medoid_Ws = [all_w[idx] for idx in medoid_indices]

for i, W in enumerate(medoid_Ws):
    print(f"Index: {medoid_indices[i]}, MSE: {medoid_mses[i]:.6f}")
    print(pd.DataFrame(W))
    print("\n")
#-------------------------Centrality--------------------------------------

def compute_centralities(W):
    n = W.shape[0]
    G = nx.DiGraph()
    
    for i in range(n):
        for j in range(n):
            if W[i, j] != 0:
                G.add_edge(i, j, weight=abs(W[i, j]))
    
    in_degree = np.zeros(n)
    for i in range(n):
        in_degree[i] = sum(abs(W[j, i]) for j in range(n) if W[j, i] != 0)
    
    out_degree = np.zeros(n)
    for i in range(n):
        out_degree[i] = sum(abs(W[i, j]) for j in range(n) if W[i, j] != 0)
    
    total_degree = in_degree + out_degree
    betweenness = list(nx.betweenness_centrality(G, weight='weight').values())
    
    try:
        eigenvector = list(nx.eigenvector_centrality(G, weight='weight', max_iter=1000).values())
    except:
        eigenvector = total_degree / np.sum(total_degree) if np.sum(total_degree) > 0 else np.ones(n)/n
    
    return {
        'in_degree': in_degree,
        'out_degree': out_degree, 
        'total_degree': total_degree,
        'betweenness': betweenness,
        'eigenvector': eigenvector
    }

best_W = all_results_by_activation['exponential']['best_results']['continuous']['W']
centrality_results = compute_centralities(best_W)

df_centrality = pd.DataFrame({
    'Node': labels,
    'In_Degree': centrality_results['in_degree'],
    'Out_Degree': centrality_results['out_degree'],
    'Total_Degree': centrality_results['total_degree'],
    'Betweenness': centrality_results['betweenness'],
    'Eigenvector': centrality_results['eigenvector']
})

df_centrality['Gender'] = [
    'Female' if 'female' in str(lbl).lower() else
    'Male' if 'male' in str(lbl).lower() else 'NA'
    for lbl in df_centrality['Node']
]

df_centrality.to_csv('centrality_table.csv', index=False)




