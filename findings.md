# Assignment 7: Analytical Findings & Statistical Interpretation

## From Visualization to Evidence

The research goal is to explain relationships among vehicle characteristics and predict manufacturer suggested retail price (MSRP). The existing project’s original Cooper Union Car Dataset contains 11,914 rows. Removing 715 exact duplicate rows and excluding missing or nonpositive target prices leaves 11,199 records. No price outliers are removed. Findings 1–4 use all eligible records descriptively; their p-values are exploratory. Trims within model families are related, so conventional row-level p-values likely overstate precision. Effect sizes and plots receive more emphasis than significance.

## Experimental design

The predictors are Year, Engine HP, Engine Cylinders, Make, Vehicle Size, Vehicle Style, Transmission Type, and Driven_Wheels. The target is MSRP. A fixed seed of 42 creates a group holdout: 8,875 training rows across 742 make/model families and 2,324 test rows across 186 different families. The split reserves roughly 20% of families, not necessarily 20% of rows. Three-fold GroupKFold on training data selects the winner by mean MAE. Median numeric imputation, missing-value indicators, standardization, most-frequent categorical imputation, and one-hot encoding are fitted inside each fold. All 16 regressors train on log(1 + MSRP), then predictions are converted back to dollars. The median baseline is additional and does not count among the 16 models. No hyperparameters are tuned using test results. Model settings appear in analysis.py and the notebook.

## 1. Horsepower and MSRP

**Research Question:** Is engine horsepower associated with MSRP?

**Evidence:** Spearman rho = 0.833, p < 0.001, n = 11,130 complete records.

![1. Horsepower and MSRP](results/horsepower.png)

**Analysis:** The rank correlation measures a monotonic relationship and is less sensitive to extreme prices than a raw-price Pearson correlation. The scatterplot uses a log price axis so lower-priced vehicles remain visible.

**Interpretation:** Higher horsepower is associated with higher listed MSRP in this sample. This supports horsepower as a price predictor. Brand, luxury equipment, and vehicle type could explain part of the relationship. Horsepower alone does not establish a causal price effect.

## 2. Price differences by vehicle size

**Research Question:** Does MSRP differ among compact, midsize, and large vehicles?

**Evidence:** Compact: median $23,650, n=4,395; Large: median $39,970, n=2,683; Midsize: median $33,295, n=4,121. Kruskal–Wallis H = 1519.16, p < 0.001.

![2. Price differences by vehicle size](results/size.png)

**Analysis:** The boxplot shows medians and spread; individual outlier points are hidden for readability but remain in all calculations. Kruskal–Wallis tests whether group distributions differ. It does not prove every pair differs or isolate a size effect.

**Interpretation:** Vehicle size is associated with price differences and is a useful candidate predictor. Differences in brand and equipment mix prevent attributing the full price gap to size. Pairwise claims would require post-hoc tests.

## 3. Horsepower and fuel economy

**Research Question:** Is higher horsepower associated with lower highway fuel economy?

**Evidence:** Spearman rho = -0.510, p < 0.001, n = 11,130 complete records.

![3. Horsepower and fuel economy](results/mpg.png)

**Analysis:** The scatterplot shows the direction and spread of the association. Some unusually high mileage values could reflect alternative-powertrain vehicles or different efficiency reporting and should be checked before drawing powertrain-specific conclusions.

**Interpretation:** Higher horsepower is associated with lower highway MPG overall in this sample. This suggests a performance–efficiency tradeoff, but it does not prove increasing horsepower causes a specific MPG loss. Fuel economy is examined descriptively and is excluded from the eight-feature MSRP benchmark.

## 4. MSRP across model years

**Research Question:** How does the distribution of MSRP vary by model year?

**Evidence:** Model years span 1990–2017. Median MSRP is $2,000 in 1990 and $36,720 in 2017. Annual sample counts are shown below the median line.

![4. MSRP across model years](results/years.png)

**Analysis:** This is a cross-sectional comparison by model year, not a longitudinal study. The available makes, models, equipment, and number of observations vary across years. Prices are not adjusted for inflation. Older observations with low or repeated prices require source validation.

**Interpretation:** The sample shows a model-year pattern in listed MSRP. It does not measure annual market inflation, depreciation of one car, or a forecast. Year is included as a predictive feature, while the interpretation remains limited to this dataset.

## 5. Comparing 16 machine learning models

**Research Question:** Which tested model predicts MSRP best for held-out make/model families?

**Evidence:** Extra Trees has the lowest training cross-validation MAE ($7,780; fold SD $747). On the held-out test set: MAE $5,575, RMSE $13,431, R² 0.922. Median-baseline test MAE is $19,728. The selected model reduces test MAE by 71.7% relative to that baseline.

![5. Comparing 16 machine learning models](results/models.png)

**Analysis:** Models are ranked by mean training cross-validation MAE, not by test results. The same group folds, features, and test set are used for all models. Fold standard deviation describes variability across three folds; it is not a confidence interval. MAE reflects typical absolute dollar error; RMSE emphasizes large errors; R² compares squared error with a test-mean reference.

**Interpretation:** The selected model is the strongest candidate under this experiment’s MAE criterion. Extra Trees averages randomized decision trees, which can capture nonlinear price relationships and interactions among brand, horsepower, and vehicle type. Averaging may reduce variance relative to a single tree, although this experiment does not isolate that mechanism. Random Forest has a lower test RMSE and higher test R², so Extra Trees does not lead on every metric. Compare its training error, fold variability, and RMSE before deployment. This is a benchmark with fixed settings, not exhaustive tuning or proof of universal superiority.

## Model performance

model,cv_mae,cv_sd,test_mae,test_rmse,test_r2,train_mae,seconds
Extra Trees,7779.853,746.55,5575.141,13431.45,0.922,2269.479,4.363
Histogram Gradient Boosting,8099.232,430.296,5912.286,14665.978,0.907,4012.223,2.491
Bagging Trees,8134.556,452.278,5864.204,12870.371,0.928,2284.23,4.561
Random Forest,8306.027,424.298,5806.736,12497.757,0.932,2708.728,4.122
Gradient Boosting,8578.488,469.384,7232.525,17493.094,0.868,5876.194,4.271
K Nearest Neighbors,10124.906,694.712,8282.687,17652.196,0.865,4585.517,0.628
RBF Support Vector,10179.49,835.9,7485.64,15707.855,0.893,3088.525,13.385
Decision Tree,10773.265,447.096,7874.289,21201.051,0.806,4782.998,0.171
AdaBoost,12432.111,1231.297,10631.92,23076.854,0.77,11644.227,3.651
Ridge,14575.079,854.348,12089.406,25692.244,0.715,12168.279,0.122
Elastic Net,14964.406,841.005,12278.642,26466.715,0.697,12314.01,0.162
Lasso,14992.308,1109.27,12642.104,27996.582,0.661,12464.439,0.142
Linear Support Vector,15530.492,1779.828,10052.136,20633.04,0.816,9973.511,2.169
Bayesian Ridge,15795.341,274.19,11055.175,20733.338,0.814,11981.603,0.236
Huber,15915.648,1567.433,10162.316,20258.292,0.823,10498.608,1.016
Linear Regression,16064.729,511.245,10978.456,20515.669,0.818,11918.005,0.161


## Limitations

Limitations: This convenience dataset is not a random sample of today’s vehicle market. MSRP is not a transaction price. Related trims, uneven brand coverage, extreme prices, model-year composition, and possible legacy price conventions limit generalization. Group holdout reduces model-family leakage but does not test unseen manufacturers or future years. Log-target fitting prioritizes relative fit and can underpredict high prices; dollar MAE and RMSE expose that limitation. Results depend on a single held-out split and fixed model settings. No causal claims or production-readiness claims are supported.

## Results-section foundation

Together, these findings support using engine characteristics, vehicle categories, and model year to study price differences. Horsepower and size show descriptive associations with MSRP, and the horsepower–MPG relationship adds an efficiency dimension. Model-year results require caution because the sample composition changes. Among the 16 tested algorithms, Extra Trees has the best training group cross-validation MAE and achieves a held-out MAE of $5,575. These results provide a foundation for the research paper’s Results section, followed by a Discussion of confounding, sample coverage, and generalization.

## Source

Cooper Union, Car Features and MSRP: https://www.kaggle.com/datasets/CooperUnion/cardataset . Data copied from the existing project archive. Raw CSV SHA-256: 26e39d3e902246d01a93ae390f51129a288079aefad2cb3292751a262ffd62d8