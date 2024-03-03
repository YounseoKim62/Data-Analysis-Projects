set.seed(777)

#loading libraries
library(data.table)
library(caret)
library(Metrics)
library(glmnet)
library(plotmo)
library(lubridate)
library(xgboost)

#importing data
test_emb<-fread('./volume/data/raw/test_emb.csv')
bind_train<-fread('./volume/data/interim/bind_train.csv')
bind_test<-fread('./volume/data/interim/bind_test.csv')
example_submission<-fread('./volume/data/raw/example_sub.csv')

#save reddit variable because dummyVars will remove
train_y <- bind_train$reddit
test_y <- bind_test$reddit

#creating dummy variables
dummies <- dummyVars(reddit ~ ., data = bind_train)
train_x <- predict(dummies, newdata = bind_train)
test_x <- predict(dummies, newdata = bind_test)

##########################
#         xgboost        #
##########################

#model preparation
#converts data into a format suitable for XGBoost training
dtrain_data <- xgb.DMatrix(train_x,label=train_y,missing=NA)
dtest_data <- xgb.DMatrix(test_x,missing=NA)

hyper_perm_tune <- NULL

#cross validation
#to determine the best number of boosting rounds
param <- list(  objective           = 'multi:softprob',
                gamma               =0.00,
                booster             = 'gbtree',
                eval_metric         = 'mlogloss',
                eta                 = 0.02,
                max_depth           = 15,
                min_child_weight    = 1,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                tree_method = 'hist',
                num_class = 11
)

XGBm <- xgb.cv(params=param, nfold=10, nrounds=10000, missing=NA,
               data=dtrain_data, print_every_n = 1, early_stopping_rounds=25)

#records the best iteration and test error
best_ntrees<-unclass(XGBm)$best_iteration
new_row<-data.table(t(param))
new_row$best_ntrees<-best_ntrees
test_error<-unclass(XGBm)$evaluation_log[best_ntrees,]$test_rmse_mean
new_row$test_error<-test_error
hyper_perm_tune<-rbind(new_row,hyper_perm_tune)

#fitting the model to all of the data

#check the evaluation metric of the model for the current number of trees.
watchlist <- list(train = dtrain_data)

#fitting the full model
XGBm<-xgb.train(params=param,nrounds=best_ntrees,missing=NA,data=dtrain_data,watchlist=watchlist,print_every_n=1)

#change test_emb into matrix
#before that we have to delete reddit column in bind_test for making a prediction
#bind_test$reddit <- NULL
#matrix_test<-as.matrix(bind_test)

#make a prediction 
matrix_test <- as.matrix(test_emb)
pred <- predict(XGBm, newdata = matrix_test)
matrix_pred <- matrix(pred, ncol = 11, byrow=TRUE)
pred < -data.table(matrix_pred)

#prepare submission
pred$id <- example_submission$id
pred <- setcolorder(pred, c('id','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11'))
example_submission$redditcars <- pred$V1
example_submission$redditCFB<- pred$V2
example_submission$redditCooking<- pred$V3
example_submission$redditMachineLearning<- pred$V4
example_submission$redditmagicTCG<- pred$V5
example_submission$redditpolitics<- pred$V6
example_submission$redditRealEstate<- pred$V7
example_submission$redditscience<- pred$V8
example_submission$redditStockMarket<- pred$V9
example_submission$reddittravel<- pred$V10
example_submission$redditvideogames<- pred$V11

#make a submission file
fwrite(example_submission,'./volume/data/processed/submit.csv')
