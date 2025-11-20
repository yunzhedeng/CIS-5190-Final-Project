from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class Model:
    def __init__(self):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression(max_iter=300))
        ])

        params = {
            'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
            'tfidf__max_features': [5000, 10000, None],
            'clf': [
                LogisticRegression(max_iter=300),
                LinearSVC()
            ]
        }

        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

    def fit(self, X, y):
        print("Start training...")
        self.model.fit(X, y)
        print("Best parameters:", self.model.best_params_)

    def predict(self, X):
        return self.model.predict(X)
