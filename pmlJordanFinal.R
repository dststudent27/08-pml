# Coursera Practical Machine Learning Peer Writeup
# pml project April 2016

# display date & time
print(currentDate <- date())

# Executive Summary
# Participants in the Human Activity Recognition (HAR) Project were asked to
# perform various exercises correctly and incorrectly in 5 different ways.Using 
# performance data collected from accelerometers fed by multiple quantified self 
# movement devices, the goal of this project is to train a model that could be 
# used to predict the manner in which the participants performed the exercises.

# The data for this project from this source: http://groupware.les.inf.puc-rio.br/har.  
# The training data here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
# The test data here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  


# Preliminaries
# Set Seed for Reproducibility
# Load required libraries
set.seed(212061996)
library(caret)
library(gmodels)
library(randomForest)

# Load the Data
trainChunk <- read.csv("pmlTrain.csv", header = TRUE, stringsAsFactors = FALSE, 
                       sep = ",", na.strings = c("NA", "", "#DIV/0!"))

testChunk <- read.csv("pmlTest.csv", header = TRUE, stringsAsFactors = FALSE, 
                      sep = ",", na.strings = c("NA", "", "#DIV/0!"))

trainChunk$classe <- as.factor(trainChunk$classe)

# Preprocess (Examine & Clean) the Data
# Remove missing values, irrelevant columns of data, and other items from the 
# data set that do not contribute to the scope of the project

# summary(trainChunk)
# str(trainChunk)
dim(trainChunk)

# remove the rowID column from the data set
removeIDCol <- trainChunk[, -1]
processedTrainChunk <- removeIDCol
dim(processedTrainChunk)

# find & remove missing values from training data
NAs <- apply(processedTrainChunk, 2, function(x) {sum(is.na(x))})
removeNAs <- processedTrainChunk[, which (NAs == 0)]
processedTrainChunk <- removeNAs
dim(processedTrainChunk)

# remove nzv (near zero variance) values
removeNZV <- nearZeroVar(processedTrainChunk)
processedTrainChunk <- processedTrainChunk[, -removeNZV]
dim(processedTrainChunk)

# remove useless predictors (features)
uselessPredictors <- grep("cvtd_timestamp|X|user_name|num_window", names (trainChunk))
processedTrainChunk <- processedTrainChunk[, -uselessPredictors]
dim(processedTrainChunk)


# Partition Training Data into Training & Validating Data Subsets
# Create Training and Validating Data Sets
inTrain <- createDataPartition(y = trainChunk$classe, p = 0.25, list = FALSE)
training <- processedTrainChunk[inTrain, ]
dim(training)
# create the test data for cross validation
validating <- processedTrainChunk[-inTrain, ] 
dim(validating)


# Modeling
# Fit the model using the Random Forest Algorithm
ctrl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
myModel <- train(training$classe ~ ., data = training, method = "rf", 
                  prof = TRUE, trControl = ctrl)
myModel


# Evaluate the Model
cvPrediction <- predict(myModel, newdata = validating)
confusionMatrix(cvPrediction, validating$classe)


accuraccy <- c(as.numeric(cvPrediction == validating$classe))
accuraccy <- sum(accuraccy) * 100/nrow(validating)
oosError <- 100 - accuraccy

# Using CrossTable (gmodels package) Provides a more Detailed Confusion Matrix
CrossTable(cvPrediction, validating$classe)


# Predict on Test Data & Write to File.
tcPrediction <- predict(myModel, testChunk, type = "raw") 
tcPrediction

pml_write_files = function(x){
      n = length(x)
      for(i in 1:n) {
            filename = paste0("problem_id_", i, ".txt")
            write.table(x[i], file = filename, quote = FALSE, row.names = FALSE,
                        col.names = FALSE)
      }
}

pml_write_files(tcPrediction)


# Conclusion
# The kappa statistic ranges from 0 to 1, inclusive, with 1 indicating perfect  
# agreement between the model's prediction and the true values.  
# Though the interpretation can be subjective, generally, a good agreement 
# typically ranges between 0.60 - 0.80.

#The model accuraccy: 
round(accuraccy, 2)  
#The out-of-sample error: 
round(oosError, 2)  
#The kappa value is 0.97.  


print(currentDate <- date())
