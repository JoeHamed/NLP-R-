# Natural Language Processing

# Importing the dataset

dataset = read.csv('Restaurant_Reviews.tsv', sep = '\t', quote = '', stringsAsFactors = FALSE) 
# or 
dataset = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE) 
# StringAsFactors --> Ensures that strings in the data are not converted to factors (useful for text processing).

# Cleaning the texts

# install.packages('tm')
library(tm) # Text Mining
corpus =  VCorpus(VectorSource(dataset$Review)) # the column that contains the text that we want to clean
#  as.character(corpus[[1]])  ---> convert an object into a character vector.
corpus = tm_map(corpus, content_transformer(tolower)) # Transformations on Corpora (corpus) -- transform to lower case
corpus = tm_map(corpus, removeNumbers) # Remove any number in the review
corpus = tm_map(corpus, removePunctuation) # Remove punctuation in the review
# install.packages('SnowballC')
library(SnowballC)
corpus = tm_map(corpus, removeWords, stopwords()) # Remove the non relevant words
# Apply Stemming
corpus = tm_map(corpus, stemDocument)
# Remove the extra spaces
coprus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model

dtm = DocumentTermMatrix(corpus) # Sparse matrix of features -- sparsity is 100%
# Filtering the irrelevant words
dtm = removeSparseTerms(dtm, sparse = 0.999) # terms appearing in less than 0.1% of the documents will be removed

# Decision Tree Classification

# converting a Document-Term Matrix (DTM) into a data frame
df = as.data.frame(as.matrix(dtm)) # as.matrix converts the Document-Term Matrix (dtm) into a standard matrix format.
# add the dependent variable to df
df$Liked = dataset$Liked #create a new column named Liked

# Ensuring that Liked is treated as a categorical variable rather than as continuous numeric data 
df$Liked = factor(df$Liked, levels = c(0, 1)) 

# Splitting the dataset into the training and test set
library(caTools)
set.seed(123)

split = sample.split(df$Liked, SplitRatio = 0.8)
training_set = subset(df, split == TRUE )
test_set = subset(df, split == FALSE)

# Fitting the classifier to the training set
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 100)

# Predicting the test results 
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')

# Making the confusion matrix
cm = table(test_set[, 692], y_pred)