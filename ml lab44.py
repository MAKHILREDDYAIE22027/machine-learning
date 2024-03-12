import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

file_path = r"C:/Users/akhil/Downloads/DATA/DATA/code_comm.csv"
data = pd.read_csv(file_path)


feature_name = input("Enter the name of the feature you want to analyze: ")
target_class_name = input("Enter the name of the target class column: ")


feature_data = data[feature_name]


plt.hist(feature_data, bins=10, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of {}'.format(feature_name))
plt.show()


mean = np.mean(feature_data)
variance = np.var(feature_data)

print("Mean:", mean)
print("Variance:", variance)

class_means = data.groupby(target_class_name)[feature_name].mean()
class_variances = data.groupby(target_class_name)[feature_name].var()

print("\nClass-wise Mean:")
print(class_means)
print("\nClass-wise Variance:")
print(class_variances)


vector1 = data.iloc[0][feature_name]  
vector2 = data.iloc[1][feature_name]  


r_values = range(1, 11)
distances = [minkowski([vector1], [vector2], r) for r in r_values]


plt.plot(r_values, distances)
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r')
plt.show()