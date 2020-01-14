import numpy as np
from scipy.stats import rankdata

# Open data file
data_file = open("data.txt", "r")

# Construction of the decision matrix...
Dec_mat = []
for i in data_file:
    x = i.split("	")
    Dec_mat.append(x)
Dec_mat = np.array(Dec_mat)
Opt_list = Dec_mat[0]
Criteria_list = Dec_mat[1]
Alternatives_list = Dec_mat[:, 0]
Dec_mat = np.delete(Dec_mat, 0, 0)
Dec_mat = np.delete(Dec_mat, 0, 0)
Dec_mat = np.delete(Dec_mat, 0, 1)
Dec_mat = np.array(Dec_mat, dtype=float)
Dec_mat_backup = np.array(Dec_mat, dtype=float)

num_alternatives = int(len(Dec_mat))
num_criteria = int(len(Dec_mat[0]))
# Negative and zero correction transformation for logarithm.
count = 0
for j in range(num_criteria):
    if min(Dec_mat_backup[:, j]) <= 0:
        count = count + 1
        min_ = min(Dec_mat_backup[:, j])
        max_ = max(Dec_mat_backup[:, j])
        for i in range(num_alternatives):

            Dec_mat_backup[i][j] = (Dec_mat_backup[i][j] - min_) / (max_ - min_)

            if Dec_mat_backup[i][j] == 0:
                Dec_mat_backup[i][j] = 0.00001
if count > 0:
    Dec_mat_normal = Dec_mat_backup
else:
    Dec_mat_normal = np.array(Dec_mat, dtype=float)
# Normalisation
for j in range(num_criteria):
    sum_ = 0
    for i in range(num_alternatives):
        sum_ = sum_ + float(Dec_mat_normal[i][j])

    for i in range(num_alternatives):
        Dec_mat_normal[i][j] = float(Dec_mat_normal[i][j]) / sum_

# Calculation of the entropy values
Entropy = [0] * num_criteria
Entropy = np.array(Entropy, dtype=float)
Diver = []
for j in range(num_criteria):

    for i in range(num_alternatives):
        Entropy[j] = Entropy[j] + (-1 / np.log(num_alternatives)) * np.log(Dec_mat_normal[i][j]) * Dec_mat_normal[i][j]
    # Calculation of the weights
    Diver.append(1 - Entropy[j])
Weight = []
for j in range(num_criteria):
    Weight.append(Diver[j] / np.sum(Diver))

# Normalization of the decision matrix
sum_matrix = []
sum_matrix = np.array(sum_matrix, dtype=float)
sum_matrix = np.sum(Dec_mat ** 2, axis=0)  # sums of squares
sum_matrix = np.array(sum_matrix ** 0.5)  # square roots of the sums
Nor_Dec_mat = np.array(Dec_mat, dtype=float)  # the normal decision matrix
Weight_Nor_Dec_mat = np.array(Dec_mat, dtype=float)  # the weighted normal decision matrix

for j in range(num_criteria):
    Nor_Dec_mat[:, j] = Dec_mat[:, j] / sum_matrix[j]  # calculation of the normal decision matrix
    Weight_Nor_Dec_mat[:, j] = Nor_Dec_mat[:, j] * Weight[j]
# Construction of the ideal solution and the anti ideal solution
Ideal_sol = []
Ideal_sol = np.array(Ideal_sol, dtype=float)

Anti_Ideal_sol = []
Anti_Ideal_sol = np.array(Anti_Ideal_sol, dtype=float)

Ideal_sol = np.amax(Weight_Nor_Dec_mat, axis=0)
Anti_Ideal_sol = np.amin(Weight_Nor_Dec_mat, axis=0)

for j in range(num_criteria):
    if Opt_list[j] == "Min":
        a = Ideal_sol[j]
        b = Anti_Ideal_sol[j]
        Ideal_sol[j] = b
        Anti_Ideal_sol[j] = a
# Calculating the separation measures
Positive_separation = []

Negative_separation = []

for i in range(num_alternatives):
    backup_pos = 0
    backup_neg = 0
    for j in range(num_criteria):
        backup_pos = backup_pos + (Weight_Nor_Dec_mat[i][j] - Ideal_sol[j]) ** 2
        backup_neg = backup_neg + (Weight_Nor_Dec_mat[i][j] - Anti_Ideal_sol[j]) ** 2
    Positive_separation.append(backup_pos ** 0.5)
    Negative_separation.append(backup_neg ** 0.5)

Positive_separation = np.array(Positive_separation, dtype=float)
Negative_separation = np.array(Negative_separation, dtype=float)

# Calculate the relative closeness to the ideal solution
Relative_closeness = []
Relative_closeness = np.array(Negative_separation, dtype=float)
for i in range(num_alternatives):
    Relative_closeness[i] = Negative_separation[i] / (Positive_separation[i] + Negative_separation[i])
# Ranking the preference the descending order of the relative closeness
Rank_Alternative = rankdata(Relative_closeness)
Rank_Alternative = num_alternatives - Rank_Alternative + 1
Rank_Alternative = np.array(Rank_Alternative, dtype=int)

# Printing the output

open('output.txt', 'w').close()
output_file = open("output.txt", "a")

output_file.write("------------The Criteria Weights-------------\n\n")
for i in range(num_criteria):
    output_file.write(" W" + "(" + str(i + 1) + ")=" + "\t" + str(Weight[i]) + "\n")
output_file.write("\n\n")

output_file.write("------------The Decision Matrix-------------\n\n")
for i in range(num_alternatives):
    for j in range(num_criteria):
        output_file.write(str(Dec_mat[i][j]) + "\t")
    output_file.write("\n")
output_file.write("\n\n")

output_file.write("------------The Normalised Decision Matrix-------------\n\n")
for i in range(num_alternatives):
    for j in range(num_criteria):
        output_file.write(str(Nor_Dec_mat[i][j]) + "\t")
    output_file.write("\n")
output_file.write("\n\n")

output_file.write("------------The Weighted Normalised Decision Matrix-------------\n\n")
for i in range(num_alternatives):
    for j in range(num_criteria):
        output_file.write(str(Weight_Nor_Dec_mat[i][j]) + "\t")
    output_file.write("\n")
output_file.write("\n\n")

output_file.write("------------The Ideal Solution-------------\n\n")
output_file.write(" V*=( ")
for i in range(num_alternatives):
    output_file.write(str(Ideal_sol[i]) + "\t")
output_file.write(")")
output_file.write("\n\n")

output_file.write("------------The Anti-Ideal Solution-------------\n\n")
output_file.write(" V-=( ")
for i in range(num_alternatives):
    output_file.write(str(Anti_Ideal_sol[i]) + "\t")
output_file.write(")")
output_file.write("\n\n")

output_file.write("------------The Separation from the Ideal -------------\n\n")
output_file.write(" S*=( ")
for i in range(num_alternatives):
    output_file.write(str(Positive_separation[i]) + "\t")
output_file.write(")")
output_file.write("\n\n")

output_file.write("------------The Separation from the Anti-Ideal -------------\n\n")
output_file.write(" S-=( ")
for i in range(num_alternatives):
    output_file.write(str(Negative_separation[i]) + "\t")
output_file.write(")")
output_file.write("\n\n")

output_file.write("------------The Relative Closeness to the Ideal Solution -------------\n\n")
output_file.write(" C*=( ")
for i in range(num_alternatives):
    output_file.write(str(Relative_closeness[i]) + "\t")
output_file.write(")")
output_file.write("\n\n")

output_file.write("------------The Rankings -------------\n\n")
output_file.write(" Alternative " + "\t" + "\t" + "Rank"+"\n")
for i in range(num_alternatives):
    output_file.write("    "+str(Alternatives_list[i+2]) + "\t" + "\t" + " " + str(Rank_Alternative[i])+"\n")

output_file.write("\n\n")


output_file.close()
