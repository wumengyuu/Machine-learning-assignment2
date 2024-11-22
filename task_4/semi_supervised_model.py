from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task_4.frozen_transformer import FrozenTransformer

# Impute missing values with the best imputers from Task 3.1
imp = SimpleImputer()
# X is the union of the unsupervised and (train) supervised feature datasets
X = imp.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)
# Try different numbers of components.
iso = Isomap(n_components=2)
iso.fit(X)

pipe = make_pipeline(SimpleImputer(),
                     scaler,
                     FrozenTransformer(iso), # <- Here is the Frozen Isomap
                     LinearRegression())

# (X_train, y_train) is the labeled, supervised data
pipe.fit(X_train, y_train)