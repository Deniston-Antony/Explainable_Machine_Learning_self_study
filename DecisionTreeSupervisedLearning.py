from sklearn.tree import DecisionTreeClassifier , export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# Land cover data from your slides: [red, near-infrared]
X = [
    [0.8, 0.2], [0.9, 0.3], [0.7, 0.1],   # urban
    [0.8, 0.8], [0.9, 0.9], [0.7, 0.7],   # bare ground
    [0.1, 0.1], [0.2, 0.2], [0.1, 0.3],   # water
    [0.2, 0.8], [0.1, 0.9], [0.3, 0.9],   # vegetation
]
y = ["urban"] * 3 + ["bare ground"] * 3 + ["water"] * 3 + ["vegetation"] * 3 # creates 3 labels for each class

model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X,y)

print(export_text(model, feature_names=["red", "near-infreared"]))  # Print the decision tree structure  
print(model.predict([[0.15,0.85]]))  # Predicting the land cover type for a new sample
print(dict(zip(["red", "near-infrared"], model.feature_importances_)))  # Feature importance scores

plot_tree(model, feature_names=["red","near-infrared"],class_names=model.classes_, filled=True)
plt.show()