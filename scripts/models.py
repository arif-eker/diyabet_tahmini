from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

gbm_params = {"learning_rate": [0.01, 0.1, 0.001],
              "max_depth": [3, 5, 8, 10],
              "n_estimators": [200, 500, 1000],
              "subsample": [1, 0.5, 0.8]}

lgbm_params = {"learning_rate": [0.01, 0.001, 0.5, 1],
               "n_estimators": [200, 500, 1000],
               "max_depth": [3, 5, 8, 10],
               "colsample_bytree": [1, 0.5, 0.4, 0.7]}

rf_params = {"max_depth": [5, 10, None],
             "max_features": [2, 5, 10],
             "n_estimators": [100, 500, 900],
             "min_samples_split": [2, 10, 30]}

xgb_params = {"learning_rate": [0.1, 0.01, 1],
              "max_depth": [2, 5, 8],
              "n_estimators": [100, 500, 1000],
              "colsample_bytree": [0.3, 0.6, 1]}


def get_tuned_models(x_train, y_train, rnd_state):

    gbm_model = GradientBoostingClassifier(random_state=rnd_state)
    lgb_model = LGBMClassifier(random_state=rnd_state)
    rf_model = RandomForestClassifier(random_state=rnd_state)
    xgb_model = XGBClassifier(random_state=rnd_state)

    gbm_cv_model = GridSearchCV(gbm_model,
                                gbm_params,
                                cv=10,
                                n_jobs=-1,
                                verbose=2).fit(x_train, y_train)

    lgbm_cv_model = GridSearchCV(lgb_model,
                                 lgbm_params,
                                 cv=10,
                                 n_jobs=-1,
                                 verbose=2).fit(x_train, y_train)

    rf_cv_model = GridSearchCV(rf_model,
                               rf_params,
                               cv=10,
                               n_jobs=-1,
                               verbose=2).fit(x_train, y_train)

    xgb_cv_model = GridSearchCV(xgb_model,
                                xgb_params,
                                cv=10,
                                n_jobs=-1,
                                verbose=2).fit(x_train, y_train)

    gbm_tuned = GradientBoostingClassifier(**gbm_cv_model.best_params_, random_state=rnd_state).fit(x_train, y_train)

    lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_, random_state=rnd_state).fit(x_train, y_train)

    rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_, random_state=rnd_state).fit(x_train, y_train)

    xgb_tuned = XGBClassifier(**xgb_cv_model.best_params_, random_state=rnd_state).fit(x_train, y_train)

    return gbm_tuned, lgbm_tuned, rf_tuned, xgb_tuned
