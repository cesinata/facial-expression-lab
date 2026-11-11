# Study 2: class imbalance (simulated)

The research placement analysed student expression data from an e-learning platform. That data is lab-owned and not public, and no code from that project is reused here.

## What is simulated
`fel/imbalance.py` generates 120 users with 100 rows each, 20 features, seven expression classes, per-user offsets and one majority class near 92%. The shares follow the shape of the distribution table in the research report (one class dominating, the rest small), not its exact values. Nothing here is the lab's data and the results are labelled simulated.

## What is compared
k-NN (k=9), logistic regression and a linear SVM, each with no fix, class weights (where supported) or oversampling of training rows only. Evaluation is user-grouped 5-fold. The tables report accuracy and macro F1, next to the accuracy of always predicting the majority class.

## Reported numbers from the original work
Accuracy 0.914, 0.922 and 0.921 and macro F1 0.250, 0.272 and 0.243 (k-NN, logistic regression, linear SVM), as reported in the research report and conference paper. They are listed in the README with the label "reported (paper)". The simulated study does not reproduce them and does not claim to.
