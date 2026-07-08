"""L1-regularized sparse linear probes.

One-vs-rest logistic regression over failure classes using
``sklearn.linear_model.LogisticRegression(penalty='l1', solver='liblinear')``,
sweeping C over {0.001, 0.01, 0.1, 1.0}. Reports accuracy, number of active
directions, and sparsity ratio per layer/component.

Status: NOT IMPLEMENTED — Phase 2. See spec.md.
"""
