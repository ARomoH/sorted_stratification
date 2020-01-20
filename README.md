# sorted_stratification
When performing a grid search on sklearn, we have several mechanisms for classification problems to stratify the sample. However, there are not too many utilities for regression problems. That is why I have developed a method called **Sorted stratification** inspired by the following [article link](https://scottclowe.com/2016-03-19-stratified-regression-partitions/) that allows combining this technique with the traditional K-fold method of sklearn. 

## Method definition

"*Let N denote the number of samples, y the target variable for the samples, and k the number of equally sized partitions we wish to create.*

*With sorted stratification, we first sort the samples based on their target variable, y. Then we step through each consecutive k samples in this order and randomly allocate exactly one of them to one of the partitions. We continue this floor(N/k) times, and the remaining mod(N,k) samples are randomly allocated to one of the k partitions.*"
