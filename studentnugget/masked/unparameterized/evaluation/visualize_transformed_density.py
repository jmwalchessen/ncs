import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from append_directories import *
import sys
home_folder = append_directory(2)
sys.path.append(home_folder)
from student_t_true_conditional_data_generation import *
import scipy
import sklearn
from sklearn.preprocessing import RobustScaler

minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .4
lengthscale = 1.6
df = 2
number_of_replicates = 10000
seed_value = 2344534
#(number_of_replicates, n**2)
tvectors, tmatrices = generate_student_nugget(minX, maxX, minY, maxY, n, variance, lengthscale, df, number_of_replicates,
                            seed_value)
gpvectors, gpmatrices = generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale, number_of_replicates,
                            seed_value)
tvectorscentered = tvectors - np.min(tvectors) + 500
print(np.max(tvectors))
#first entry is yeo-johnson transformed data and lambda is second entry
a = (scipy.stats.boxcox((tvectorscentered).reshape((number_of_replicates*(n**2))), lmbda = .95))
a = a - np.mean(a)
print(np.quantile(tvectors, [.01,.05,.1,.25,.5,.75,.9, .95,.99]))
print(np.quantile(a, [.01,.05,.1,.25,.5,.75,.9,.95,.99]))
a = a.reshape((number_of_replicates,n**2))
fig, ax = plt.subplots(1)
pdd = pd.DataFrame(a[:,343], columns = ["yj"])
opdd = pd.DataFrame((tvectors)[:,343], columns = ["o"])
gpdd = pd.DataFrame(gpvectors[:,343], columns = ['gp'])
sns.kdeplot(data = pdd["yj"], color = "orange", bw_adjust = 1, ax = ax)
sns.kdeplot(data = opdd["o"], color = "blue", bw_adjust = 1, ax = ax)
sns.kdeplot(data = gpdd["gp"], color = "green", bw_adjust = 1, ax = ax)
ax.set_xlim(-5,5)
ax.set_ylim(0,1)
plt.savefig("temp.png")


