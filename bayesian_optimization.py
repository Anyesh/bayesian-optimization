import warnings

warnings.filterwarnings('ignore')

import time

from numpy import mean
from rich.console import Console
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

console = Console()

X, y = make_blobs(n_samples=500, centers=3, n_features=2)
model = KNeighborsClassifier()
search_space = [Integer(1, 5, name='n_neighbors'), Integer(1, 2, name='p')]
 
@use_named_args(search_space)
def evaluate_model(**params):
	model.set_params(**params)
	result = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='accuracy')
	estimate = mean(result)
	return 1.0 - estimate

start_time = time.time()

with console.status("[bold green] Searching...") as status:
    result = gp_minimize(evaluate_model, search_space)

print(f'Total time: {time.time() - start_time} sec')
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
