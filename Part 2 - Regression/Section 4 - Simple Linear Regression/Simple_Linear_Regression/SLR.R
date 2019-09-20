# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)
mdl = lm(formula = Salary ~ YearsExperience,
         data = training_set)
pred = predict(mdl, test_set)

#install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(mdl, training_set)),
            colour = 'blue') +
  ggtitle('YearsExperience vs Salary(training set)') +
  xlab('YearsExperience')+
  ylab('Salary')

ggplot()+
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(mdl, training_set)),
            colour = 'blue') +
  ggtitle('YearsExperience vs Salary(test set)') +
  xlab('YearsExperience')+
  ylab('Salary')