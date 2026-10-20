# Viva preparation (5 + 3 marks) — every member should be able to answer ALL of these
## Classification (Part B & comparison)
1. Why stratified split? The data is 76/24 imbalanced — why is accuracy misleading, and why weighted-F1 (and what does macro-F1/recall of the >50K class tell you)?
2. What does fitting the scaler on the full data do, and why is it leakage? Where did you avoid it (Pipeline)?
3. Random Forest vs Bagging with trees — what is the extra randomisation? Why does it reduce variance?
4. Bagging vs boosting: how do AdaBoost and Gradient Boosting differ (reweighting samples vs fitting residuals/gradients)? Why did tuning give only small gains here?
5. What does MLP need that trees don't (scaling)? Which hyperparameters did you tune?
6. How is ROC-AUC computed (binary here; One-vs-Rest for multi-class)? Why use predicted probabilities, not labels?
7. Ranks 2–8 are within ~1–2 F1 points — what does this say? Why did Gaussian Naive Bayes collapse (31.8% accuracy) after one-hot encoding?
8. Why did you select the final model with CV rather than test-set score?
9. What is the bias–variance reading of a single Decision Tree scoring lowest?
10. Interpretability vs accuracy: when would you choose Logistic Regression over SVC?
## Clustering
11. The Mall data has no labels — how do you validate clusters? What does ARI between K-Means and Ward tell you?
12. How do you choose k? What do the elbow and silhouette say; do they agree? What does silhouette ≈0.43 mean? Why k=6 and not 5?
13. Explain Silhouette, Davies-Bouldin (lower better), Calinski-Harabasz (higher better) in one sentence each.
14. Why did single linkage give one huge cluster (chaining)? Why does its Davies-Bouldin look "best" — and why is that misleading?
15. K-Means assumptions (spherical, similar size, needs k, scale-sensitive). Why standardise first?
16. What does a dendrogram height mean? How do you cut it to get k clusters?
17. PCA vs t-SNE: what does each preserve? Why shouldn't you trust distances/cluster sizes in t-SNE? What % variance do 2 PCs explain?
18. Complexity: K-Means ~O(n·k·i), hierarchical O(n²) memory — which scales to a million rows?
## Pipeline / GitHub / integrity
19. Walk through the repo; show commit history; who owns which track?
21. Business: describe each customer segment and a marketing action for it.
22. Fairness: adult data has gender/race — what risks come with an income classifier?
20. What did AI help with (scaffolding) and what is your own analysis? Show where you edited the draft observations.
