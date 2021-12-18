# Fellowship.AI-Aditya-Goel
### Steps taken:
1. Cleaning the Advertizement posts in terms of punctuation and stopwords
2. Joining the Post with definition
3. dividing the data into train and test sets
4. using naive Bayes Classifier to find an accuracy of 91%

### Using Doc2Vec Model
1. Using TaggedDocument Method to label the sentences
2. Training the model on 30 epochs to get the values for vectorizer
3. Using Logistic Regression on trained Doc2Vec Features
4. Accuracy recieved 83%

### Improvements made
1. Missing data removed
2. All Classes distributions equalized
3. Punctuation marks removed

### Ways to Improve the model further
1. We will be looking for improvements from Naive Bayes as this model has proved to provide the best accuracy till now.
2. Using a correlation matrix redundant features can be removed to find a better accuracy.
3. We can try using log probabilites as due to a number of features the probability fraction might be very less initially.
4. Grid search CV can be used to find different parametric combinations that can prove to be the best fit for the Multinomial Naive Bayes
