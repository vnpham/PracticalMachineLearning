library(caret)
set.seed(12345)

tc.all <- read.csv("pml-training.csv", head=TRUE)
tc.predict <- read.csv("pml-testing.csv", head=TRUE)

tc.all <- tc.all[,colSums(is.na(tc.predict))<nrow(tc.predict)]
tc.predict <- tc.predict[,colSums(is.na(tc.predict))<nrow(tc.predict)]
tc.all  <- tc.all[,-c(1,5,6)]
tc.predict <- tc.predict[,-c(1,5,6)]

index.train <- createDataPartition(y=tc.all$classe, p=0.6, list=FALSE)
tc.train <- tc.all[index.train,]
tc.test.validate <- tc.all[-index.train,]
index.validate <- createDataPartition(y=tc.test.validate$classe, p=0.5, list=FALSE)
tc.validate <- tc.test.validate[index.validate,]
tc.test <- tc.test.validate[-index.validate,]

# rm(tc.all, tc.test.validate, index.train, index.validate)

library(doParallel)
cl <- makeCluster(6)
registerDoParallel(cl)

trControl = trainControl(method="cv", number=10)

st.gbm <- system.time(model.gbm <- train(classe ~ ., method="gbm",
                                         verbose=FALSE,
                                         data=tc.train, trControl=trControl))
st.rf <- system.time(model.rf <- train(classe ~ ., method="rf", prox=FALSE,
                                       verbose=FALSE,
                                       data=tc.train, trControl=trControl))

stopCluster(cl)

cm.gbm <- confusionMatrix(tc.validate$classe, predict(model.gbm, tc.validate))
cm.rf <- confusionMatrix(tc.validate$classe, predict(model.rf, tc.validate))

(cm.validate.rf <- confusionMatrix(tc.test$classe, predict(model.rf, tc.test)))

#------------------------------------------------
pml_write_files = function(x){
   n = length(x)
   for(i in 1:n){
      filename = paste0("problem_id_",i,".txt")
      write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
   }
}

answers <- predict(model.rf, tc.predict)
pml_write_files(answers)