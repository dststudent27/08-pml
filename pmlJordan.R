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


# Preliminaries
# Set Seed for Reproducibility
# Load required libraries
set.seed(212061996)
library(caret)

# Load the training Data
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

# determine existence of missing values (as NAs) in the training Chunk data
NAs <- apply(processedTrainChunk, 2, 
             function(x) {
                  sum(is.na(x))
                  }
            )

# remove the missing values from the training data set
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


# Define Data Partitions
# Create Training and Test Data Sets
inTrain <- createDataPartition(y = trainChunk$classe, p = 0.20, list = FALSE)
trainSubset <- processedTrainChunk[inTrain, ]
dim(trainSubset)
# create the test data for cross validation
testSubset <- processedTrainChunk[-inTrain, ] 
dim(testSubset)

# Define the Model
# Set parameters of the train control function for cross-validation
# Fit the model using the Random Forest algorithm
ctrl <- trainControl(method = "cv", number = 4, allowParallel = TRUE)
modelFit <- train(trainSubset$classe ~ ., data = trainSubset, method = "rf", 
                prof = TRUE, trControl = ctrl)
modelFit

# Display the Accuracy of the Model
resultsTR <- modelFit$results
round(max(resultsTR$Accuracy), 4) * 100

# Cross-Validation
# For the purpose of predicting values, apply the fitted model to the test data 
# set created specifically for cross-validation
cvPrediction <- predict (modelFit, testSubset)
testSubset$predRight <- cvPrediction == testSubset$classe
table(cvPrediction, testSubset$classe)

# Display the Accuracy of the Prediction 
predictedResults <- postResample(cvPrediction, testSubset$classe)
predictedResults

# Approximate Out-of-Sample Error
confMatrix <- confusionMatrix(cvPrediction, testSubset$classe)
confMatrix


# Process the Test Data
# Load the test Data
testChunk <- read.csv("pmlTest.csv", header = TRUE, stringsAsFactors = FALSE, 
                      sep = ",", na.strings = c("NA", "", "#DIV/0!"))

#testChunk$classe <- as.factor(testChunk$classe)

# remove the rowID column from the test data set
removeIDCol <- testChunk[, -1]
processedTestChunk <- removeIDCol
dim(processedTestChunk)

# determine existence of missing values (as NAs) in the training Chunk data
NAs <- apply(processedTestChunk, 2, 
             function(x) {
                   sum(is.na(x))
             }
)

# remove the missing values from the training data set
removeNAs <- processedTestChunk[, which (NAs == 0)]
processedTestChunk <- removeNAs
dim(processedTestChunk)

# remove nzv (near zero variance) values
removeNZV <- nearZeroVar(processedTestChunk)
processedTestChunk <- processedTestChunk[, -removeNZV]
dim(processedTestChunk)

# remove useless predictors (features)
uselessPredictors <- grep("cvtd_timestamp|X|user_name|num_window", names (testChunk))
processedTestChunk <- processedTestChunk[, -uselessPredictors]
dim(processedTestChunk)


# Predictions using the Test Data
tcPrediction <- predict(modelFit, testChunk, type = "raw")
tcPrediction

#testChunk$predRight <- predictionOnTestChunk == testChunk$classe
#table(predictionOnTestChunk, testChunk$classe)

pml_write_files = function(x){
      n = length(x)
      for(i in 1:n) {
            filename = paste0("problem_id_", i, ".txt")
            write.table(x[i], file = filename, quote = FALSE, row.names = FALSE,
                        col.names = FALSE)
      }
}

pml_write_files(tcPrediction)

