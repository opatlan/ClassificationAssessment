---
title: "Report"
output:
  html_document:
    df_print: paged
---

library("ggplot2")
library("skimr")
library("tidyverse")
library("mlr3")
library("mlr3learners")
library("mlr3proba")
library("data.table")
library("mlr3verse")
library("mlr3filters")
library("mlr3fselect")
library("mlr3pipelines")
library("mlr3tuning")
library("paradox")
   
#Load CSV and intial exploration of the data
 
heart.data<- read.csv("https://raw.githubusercontent.com/opatlan/ClassificationAssessment/main/heart_failure.csv")
sk.tbl<-skim(heart.data)
DataExplorer::plot_bar(heart.data, ncol = 3)
DataExplorer::plot_histogram(heart.data, ncol = 3)
   
#Start modelling and configure the task
 
set.seed(212) # set seed for reproducibility
heart.data <- heart.data %>%
  mutate(fatal_mi = as.factor(fatal_mi))
heart_task <- TaskClassif$new(id = "HeartData",
                               backend = heart.data, # <- NB: no na.omit() this time
                               target = "fatal_mi",
                               positive = "1")
   

#Define Learners and sampling strategies
 
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.12, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrn_lda <- lrn("classif.lda",predict_type = "prob")
lrn_glmnet <- lrn("classif.cv_glmnet", predict_type = "prob")

cv5 <- rsmp("cv", folds = 5)
cv10 <- rsmp("cv", folds = 10)
rb30 = rsmp("bootstrap", repeats = 30)
rb100 = rsmp("bootstrap", repeats = 100)
ho7 = rsmp("holdout", ratio = 0.7)
ho8 = rsmp("holdout", ratio = 0.8)

cv5$instantiate(heart_task)
cv10$instantiate(heart_task)
rb30$instantiate(heart_task)
rb100$instantiate(heart_task)
ho7$instantiate(heart_task)
ho8$instantiate(heart_task)
   

#Benchmark Hold Out 70-30
 
resho7 <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,lrn_ranger,
                    lrn_cart,lrn_xgboost,
                    lrn_cart_cp,lrn_log_reg,
                    lrn_lda,lrn_glmnet),
  resampling = list(ho7)
), store_models = TRUE)

resho7$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
autoplot(resho7,type = "roc")
   
#Benchmark Hold Out 80-20
 
resho8 <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,lrn_ranger,
                    lrn_cart,lrn_xgboost,
                    lrn_cart_cp,lrn_log_reg,
                    lrn_lda,lrn_glmnet),
  resampling = list(ho8)
), store_models = TRUE)

resho8$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
autoplot(resho8,type = "roc")
   


#Benchmark 5 FOld Cross Validation
 
rescv3 <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,lrn_ranger,
                    lrn_cart,lrn_xgboost,
                    lrn_cart_cp,lrn_log_reg,
                    lrn_lda,lrn_glmnet),
  resampling = list(cv5)
), store_models = TRUE)

rescv3$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
autoplot(rescv3,type = "roc")
   
#Benchmark 10 FOld Cross Validation
 
rescv10 <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,lrn_ranger,
                    lrn_cart,lrn_xgboost,
                    lrn_cart_cp,lrn_log_reg,
                    lrn_lda,lrn_glmnet),
  resampling = list(cv10)
), store_models = TRUE)

rescv10$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
autoplot(rescv10,type = "roc")
   
#Benchmark 30 Bootstrap
 
resb30 <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,lrn_ranger,
                    lrn_cart,lrn_xgboost,
                    lrn_cart_cp,lrn_log_reg,
                    lrn_lda,lrn_glmnet),
  resampling = list(rb30)
), store_models = TRUE)

resb30$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
autoplot(resb30,type = "roc")
   

#Benchmark Bootstrap 100
 
resb100 <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,lrn_ranger,
                    lrn_cart,lrn_xgboost,
                    lrn_cart_cp,lrn_log_reg,
                    lrn_lda,lrn_glmnet),
  resampling = list(rb100)
), store_models = TRUE)

resb100$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
autoplot(resb100,type = "roc")
   

#Selecting Sample Strategy

#LDA Bootstrap100 Holdout 70-30, 10 CV
 
resldasmp <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_lda),
  resampling = list(rb100,ho7,cv10)
), store_models = TRUE)

resldasmp$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
   
#Lasso Elastic Net Bootstrap100 Holdout 70-30, 10 CV
 
resglmsmp <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_glmnet),
  resampling = list(rb100,ho7,cv10)
), store_models = TRUE)

resglmsmp$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
   
#Random Forest Bootstrap100 Holdout 70-30, 10 CV
 
resrangsmp <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_ranger),
  resampling = list(rb100,ho7,cv10)
), store_models = TRUE)

resrangsmp$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
   

#Threshold Tunning
 
costs = matrix(c(-0.35, 0, 1, 0), nrow = 2)
dimnames(costs) = list(response = c(1, 0), truth = c(1, 0))
lrn_ranger=lrn("classif.ranger", predict_type = "prob", importance = "impurity")
resran = resample(heart_task, lrn_ranger, ho7,store_models = TRUE)
pran = resran$prediction()
confusion = resran$prediction()$confusion
print(confusion)
rsm_ranger <- resran$learners[[1]]$model
autoplot(resran, type = "roc")
   


 
cost_measure = msr("classif.costs", costs = costs)
missclass_measure = msr("classif.ce")
set_threshold = function(pt, th) {
  pt$set_threshold(th)
  list(confusion = pt$confusion, costs = pt$score(measures = cost_measure, task = heart_task), misclass = pt$score(measures = missclass_measure, task = heart_task))
}

set_threshold(pran, 0.6)

optfun = function(th) {
  set_threshold(pran, th)$costs
}
best = optimize(optfun, c(0.5, 1))
print(best)
set_threshold(pran, 0.691)
   
#Tuning of other hyperparameters
 
lrn_ranger$properties
filter = FilterJMIM$new()
filter$calculate(heart_task)
as.data.table(filter)
   
 
filter_cor = FilterCorrelation$new()
filter_cor$param_set
   
 
filter.ranger = flt("importance", learner = lrn_ranger)
filter.ranger$calculate(heart_task)
head(as.data.table(filter.ranger))

   

 
evals20 = trm("evals", n_evals = 20)
instance.ranger = FSelectInstanceSingleCrit$new(
  task = heart_task,
  learner = lrn_ranger,
  resampling = ho7,
  measure = cost_measure,
  terminator = evals20
)
fselector = fs("random_search")
fselector$optimize(instance.ranger)
instance.ranger$result_y
   
#LAsso Elastic Net
 
rescvglmnet = resample(heart_task, lrn_glmnet, ho7,store_models = TRUE)
rsm_cvglmnet <- rescvglmnet$learners[[1]]$model
autoplot(rescvglmnet, type = "roc")
plot(rsm_cvglmnet)
   
#Trees

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(heart_task, lrn_cart_cv, cv10, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)
tree1 <- res_cart_cv$learners[[1]]
   


 
# This is a fitted rpart object, so we can look at the model within
tree1_rpart <- tree1$model
# If you look in the rpart package documentation, it tells us how to plot the
# tree that was fitted
plot(res_cart_cv$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res_cart_cv$learners[[5]]$model, use.n = TRUE, cex = 0.8)
   

#Linear Discriminant Analysis
 
reslda = resample(heart_task, lrn_lda, cv10,store_models = TRUE)
rsm_lda <- reslda$learners[[1]]$model
autoplot(reslda, type = "roc")
   
#Fitting with tunning library
gr.ranger = lrn("classif.ranger", predict_type = "prob") %>>% po("threshold")
lran = GraphLearner$new(gr.ranger)
lran$param_set
   

#Tune paramspace
 
ps.ranger = ps(threshold.thresholds = p_dbl(lower = 0, upper = 1),
               classif.ranger.regularization.factor = p_dbl(lower = 0, upper = 1))
at.ranger = AutoTuner$new(
learner = lran,
resampling =rsmp("cv", folds = 5),
measure = cost_measure,
search_space = ps.ranger,
terminator = evals20,
tuner = TunerRandomSearch$new()
)
at.t=at.ranger$train(heart_task)
at.ranger$param_set
   
 
print(at.ranger$tuning_result$threshold.thresholds)
print(at.ranger$tuning_result$classif.ranger.regularization.factor)
   


 
resfstun = resample(heart_task, at.ranger, ho7,store_models = TRUE)
pran = resfstun$prediction()
confusion = resfstun$prediction()$confusion
print(confusion)
resfstun$score()
   


 
gr.cvglmnet = lrn("classif.cv_glmnet", predict_type = "prob") %>>% po("threshold")
lcvglmnet = GraphLearner$new(gr.cvglmnet)
lcvglmnet$param_set
   
 
ps.cvglmnet = ps(threshold.thresholds = p_dbl(lower = 0, upper = 1),
               classif.cv_glmnet.lambda.min.ratio = p_dbl(lower = 0, upper = 1))
at.cvglmnet = AutoTuner$new(
learner = lcvglmnet,
resampling =rsmp("cv", folds = 5),
measure = cost_measure,
search_space = ps.cvglmnet,
terminator = evals20,
tuner = TunerRandomSearch$new()
)
at.t=at.cvglmnet$train(heart_task)
at.cvglmnet$param_set
  
 
print(at.cvglmnet$tuning_result$threshold.thresholds)
print(at.cvglmnet$tuning_result$classif.cv_glmnet.lambda)
   

 
rescvglmnet = resample(heart_task, at.cvglmnet, ho7,store_models = TRUE)
pcvg = rescvglmnet$prediction()
confusion = rescvglmnet$prediction()$confusion
print(confusion)
rescvglmnet$score()
   

 

  