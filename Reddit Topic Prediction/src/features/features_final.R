library(data.table)
library(caret)
library(Metrics)
library(glmnet)
library(plotmo)
library(lubridate)
library(xgboost)

set.seed(777)

test_emb<-fread('./volume/data/raw/test_emb.csv')
test<-fread('./volume/data/raw/kaggle_test.csv')
train<-fread('./volume/data/raw/kaggle_train.csv')
train_emb<-fread('./volume/data/raw/train_emb.csv')
example_submission<-fread('./volume/data/raw/example_sub.csv')

#cbind train and train_emb, cbind test and test_emb
bind_train <- cbind(train, train_emb)
bind_test <- cbind(test, train_emb)

#we do not need id and text
bind_train$id <- NULL
bind_train$text <- NULL
bind_test$id <- NULL
bind_test$text <- NULL

#add reddit column with 0 values to test data
bind_test$reddit <- 0

#change reddit (categorical) to numeric value
#{0: cars, 1: CFB, 2: Cooking, 3: MachineLearning, 4: magicTCG, 5: politics, 6: RealEstate, 7: science, 8: StockMarket, 9: travel, 10: videogames}
bind_train$reddit <- as.numeric(as.factor(bind_train$reddit)) - 1

#save processed data
fwrite(bind_train,'./volume/data/interim/bind_train.csv')
fwrite(bind_test,'./volume/data/interim/bind_test.csv')

