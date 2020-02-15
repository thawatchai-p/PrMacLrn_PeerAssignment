library(caret)
library(corrplot)
library(knitr)
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
testing <- testing[ ,-c(1:7)]

training <- training[ ,colSums(is.na(training)) == 0]
testing <- testing[ ,colSums(is.na(testing)) == 0]

training <- training[, -nearZeroVar(training)]
testing  <- testing[, -nearZeroVar(testing)]

t(data.frame(Training = dim(training), Testing = dim(testing), row.names = c("Observations", "Variables")))

strip.outliers <- function(v, upperz = 3, lowerz = -3, replace = NA) 
{
        z <- scale(v) # gives z-scores for vector v
        v[!is.na(z) & (z>upperz | z<lowerz)] <- replace 
        return(v)
}
training.out.rm <- data.frame(lapply(training[,-c(53)], strip.outliers))
training <- cbind(training.out.rm, classe = training$classe)
summary(training)
training.backup <- training

cor.mat <- cor(training[, -53], use = "complete.obs"); diag(cor.mat) <- 0
which(cor.mat > 0.9, arr.ind = TRUE)
corrplot(cor.mat, order = "FPC", method = "color", type = "upper", tl.cex = 0.6, tl.col = rgb(0, 0, 0))
highlyCorrelated <- findCorrelation(cor.mat, cutoff=0.9)
names(training)[highlyCorrelated]

trControl <- trainControl(method="cv", number=5, verboseIter = FALSE)
modFit.cart <- train(classe ~ ., method="rpart", data=training, trControl=trControl, na.action = na.omit)
modFit.rf <- train(classe ~ ., method="rf", data=training, trControl=trControl, na.action = na.omit)
modFit.gbm <- train(classe ~ ., method="gbm", data=training, trControl=trControl, na.action = na.omit)

pred.cart <- predict(modFit.cart, newdata=testing)
cm.cart <- confusionMatrix(predCART, testing$classe)
pred.rf <- predict(modFit.rf, newdata=testing)
cm.rf <- confusionMatrix(predRF, testing$classe)
pred.gbm <- predict(modFit.gbm, newdata=testing)
cm.gbm <- confusionMatrix(predGBM, testing$classe)
data.frame(Model = c("CART", "RF", "GBM"), Accuracy = c(cm.cart$overall[1], cm.rf$overall[1], cm.gbm$overall[1]))

varImp(modFit.rf)

validMod <- predict(modFit.rf, newdata = validation); validMod

fancyRpartPlot(modFit.cart$finalModel)

plot(modFit.rf,main="Accuracy of Random forest model by number of predictors")

plot(modFit.gbm)
