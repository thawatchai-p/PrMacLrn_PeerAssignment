library(caret)
library(corrplot)
library(dplyr)
library(gbm)
library(ggplot2)
library(rattle)
library(randomForest)

trainURL = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(trainURL, "./training.csv")
download.file(testURL, "./testing.csv")
raw.training <- read.csv("./training.csv")
validation <- read.csv("./testing.csv")
t(data.frame(Training = dim(raw.training), Validation = dim(validation),
             row.names = c("Observations", "Variables")))

set.seed(12345)
inTrain <- createDataPartition(y = raw.training$classe, p = 0.7, list = FALSE)
training <- raw.training[inTrain, ]
testing <- raw.training[-inTrain, ]
t(data.frame(Training = dim(training), Testing = dim(testing), row.names = c("Observations", "Variables")))

training <- training[ ,-c(1:7)]
testing <- testing[,-c(1:7)]

training <- training[ ,colSums(is.na(training)) == 0]
testing <- testing[ ,colSums(is.na(testing)) == 0]

training <- training[, -nearZeroVar(training)]
testing  <- testing[, -nearZeroVar(testing)]

t(data.frame(Training = dim(training), Testing = dim(testing), row.names = c("Observations", "Variables")))

strip_outliers <- function(v, upperz = 3, lowerz = -3, replace = NA) 
{
        z <- scale(v) # gives z-scores for vector v
        v[!is.na(z) & (z>upperz | z<lowerz)] <- replace 
        return(v)
}
trainning <- data.frame(lapply(training[,-c(53)], strip_outliers))

preProcess()
cor.mat <- abs(cor(training[, -53])); diag(cor.mat) <- 0
which(cor.mat > 0.9, arr.ind = TRUE)
corrplot(cor.mat, order = "FPC", method = "color", type = "upper",
         tl.cex = 0.6, tl.col = rgb(0, 0, 0))
highlyCorrelated <- findCorrelation(cor.mat, cutoff=0.9)
names(training)[highlyCorrelated]

trControl <- trainControl(method="cv", number=5, verboseIter = FALSE)
model_cart <- train(classe ~ ., method="rpart", preProcess="knnImpute", data=training, trControl=trControl)
model_rf <- train(classe ~ ., method="rf", preProcess="knnImpute", data=training, trControl=trControl)
model_gbm <- train(classe ~ ., method="gbm", preProcess="knnImpute", data=training, trControl=trControl)

predCART <- predict(model_cart, newdata=testing)
cmCART <- confusionMatrix(predCART, testing$classe)
predRF <- predict(model_rf, newdata=testing)
cmRF <- confusionMatrix(predRF, testing$classe)
predGBM <- predict(model_gbm, newdata=testing)
cmGBM <- confusionMatrix(predGBM, testing$classe)
data.frame(Model = c("CART", "RF", "GBM"), Accuracy = c(cmCART$overall[1], cmRF$overall[1], cmGBM$overall[1]))

varImp(model_rf)

validMod <- predict(model_rf, newdata = validation); validMod

fancyRpartPlot(model_cart$finalModel)

plot(model_rf,main="Accuracy of Random forest model by number of predictors")

plot(model_gbm)
