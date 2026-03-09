import numpy as np
import torch

def hungarian_algorithm(cost_matrix):
    """
    cost_matrix: numpy array of shape (n, m)
    Return optimal row and column indices (row_ind, col_ind).
    """
    n, m = cost_matrix.shape
    size = max(n, m)

           
    padded = np.zeros((size, size))
    padded[:n, :m] = cost_matrix

           
    u = np.zeros(size)
    v = np.zeros(size)
    ind = -np.ones(size, dtype=int)

    for i in range(size):
        links = np.full(size, -1, dtype=int)
        mins = np.full(size, np.inf)
        visited = np.zeros(size, dtype=bool)
        marked_i = i
        marked_j = -1
        j = 0
        while True:
            j = -1
            for j1 in range(size):
                if not visited[j1]:
                    cur = padded[marked_i, j1] - u[marked_i] - v[j1]
                    if cur < mins[j1]:
                        mins[j1] = cur
                        links[j1] = marked_j
                    if j == -1 or mins[j1] < mins[j]:
                        j = j1
            delta = mins[j]
            for j1 in range(size):
                if visited[j1]:
                    u[ind[j1]] += delta
                    v[j1] -= delta
                else:
                    mins[j1] -= delta
            u[marked_i] += delta
            visited[j] = True
            marked_j = j
            marked_i = ind[j]
            if marked_i == -1:
                break
        while True:
            if links[j] != -1:
                ind[j] = ind[links[j]]
                j = links[j]
            else:
                break
        ind[j] = i

    row_ind = []
    col_ind = []
    for j in range(m):
        if ind[j] < n:
            row_ind.append(ind[j])
            col_ind.append(j)
    return np.array(row_ind), np.array(col_ind)


def hungarian_match(pred_concepts, true_concepts, concept2emb):
    embs = list(concept2emb.values())[0]
    
    for pred_list, true_list in zip(pred_concepts, true_concepts):
        pred_embs = [embs[concept] for concept in pred_list]
        true_embs = [embs[concept] for concept in true_list]
        
        if not pred_embs or not true_embs:
            continue
        pred_tensor = torch.stack(pred_embs)
        true_tensor = torch.stack(true_embs)
        
    if len(pred_tensor) == 0 or len(true_tensor) == 0:
        return 0.0, 0.0

                 
    pred_embs = np.array([concept2emb[i] for i in pred_tensor])
    true_embs = np.array([concept2emb[j] for j in true_tensor])

           
    sim_matrix = np.zeros((len(pred_embs), len(true_embs)))
    for i, emb_p in enumerate(pred_embs):
        for j, emb_t in enumerate(true_embs):
            sim_matrix[i, j] = np.dot(emb_p, emb_t) / (
                np.linalg.norm(emb_p) * np.linalg.norm(emb_t) + 1e-8
            )

                             
    cost_matrix = -sim_matrix
    row_ind, col_ind = hungarian_algorithm(cost_matrix)

           
    matched_pred_ids = [pred_tensor[i] for i in row_ind]
    matched_true_ids = [true_tensor[j] for j in col_ind]

                        
    correct = sum(p == t for p, t in zip(matched_pred_ids, matched_true_ids))
    matching_acc = correct / len(matched_true_ids)

             
    mean_cosine = sim_matrix[row_ind, col_ind].mean()

    return matching_acc, mean_cosine
