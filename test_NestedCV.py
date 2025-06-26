from statsmodels.stats.anova import AnovaRM
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from Path import get_full_path as gfp
from SandTwoLayer import EvaluateBy, Model, TwoLayer
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pingouin as pg

def test_model(model_name, model_instance, parameters, inner_cv,
               outer_cv_folds, x, y, groups):
    _model = Model(
        model_name = model_name,
        model_instance = model_instance,
        params = parameters
    )
    ncv = TwoLayer(
        models = [_model],
        inner = inner_cv,
        outer = outer_cv,
        X = x,
        Y = y,
        groups = groups,
        evaluate_by = EvaluateBy.ExactMatch.value,
        n_jobs = -1,
        verbose = 1
    )
    ncv.fit()
    # Get results for XGBClassifier
    results = ncv.get_results()
    print(f"Mean score for {model_name}: {results['mean_score']:.3f} ± {results['std_score']:.3f}")
    print(f"Best parameters for {model_name}: {ncv.get_best_params()}")
    return ncv

# Load the dataset
data_path = gfp("assignment/task_2/HR_data.csv")


data = pd.read_csv(data_path)
X = data.drop(columns=[i for i in data.columns if not i.startswith('HR_')])
Y = data['Frustrated'].values
#Y = np.array([i if i not in [6,7,8] else "Other" for i in Y])
Y = np.array([i if i not in [6,7,8] else 11 for i in Y])

groups = data['Individual'].values
#standardize the X
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X: np.ndarray = scaler.fit_transform(X)  # Example groups

inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = LeaveOneGroupOut()

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=0
)

params = {'C': 1 / np.logspace(0, 3, 10)}
# Define cross-validation strategies
#inner_cv = GroupKFold(n_splits=3, shuffle=True, random_state=0)


log_reg = test_model("LogReg", model, params, inner_cv, outer_cv, X, Y, groups)


base_xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=0,
)

xgb_parameters = {
    "n_estimators":   [100, 200, 300],      # trees
    "learning_rate":  [0.025, 0.05, 0.10, 0.20],    # shrinkage
    "max_depth":      [3, 5, 10],          # tree depth
    "subsample":      [0.8],           # row sampling
    "colsample_bytree":[0.8],          # feature sampling
    "reg_lambda":     np.logspace(0, 3, 4)         # L2 penalty
}

xgb = test_model("XGB", base_xgb, xgb_parameters, inner_cv, outer_cv, X, Y, groups)

baseline = DummyClassifier(strategy="most_frequent")
baseline_params = {}

baseline = test_model("Baseline", baseline, baseline_params, inner_cv, outer_cv, X, Y, groups)


score_xgb = xgb.get_scores()
test_sizes = xgb.get_test_sizes()

scores_log_reg = log_reg.get_scores()


scores_baseline = baseline.get_scores()

k = len(score_xgb)

scores = {
    'LogReg': np.array(scores_log_reg),
    'XGB':    np.array(score_xgb),
    'Baseline': np.array(scores_baseline),
}


folds = np.arange(k)
cv_df = pd.DataFrame({
    'fold':  np.tile(folds, len(scores)),
    'model': np.repeat(list(scores.keys()), k),
    'score': np.concatenate(list(scores.values())),
})

aov = AnovaRM(data=cv_df,
              depvar='score',
              subject='fold',
              within=['model'])
res = aov.fit()
print(res)



ols_model = smf.ols('score ~ C(model) + C(fold)', data=cv_df).fit()

# QQ-plot
sm.qqplot(ols_model.resid, line='s')
plt.title('QQ-plot of resediuals From ANOVA RM')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Observed Quantiles')
plt.tight_layout()
plt.savefig(gfp("Documents/Part2/qqplot.png"))
plt.show()


posthocs = pg.pairwise_tests(
    data=cv_df,
    dv='score',
    within='model',
    subject='fold',
    parametric=True,
    padjust='holm'
)
print(posthocs)
posthocs.to_latex(gfp("Documents/Part2/posthoc.tex"))