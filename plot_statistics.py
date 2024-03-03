import seaborn as sns
import matplotlib.pyplot as plt

sizes = [128, 256, 512, 1024, 8192, 32768, 65536, 131072]

values_5_cpu = [5.58, 11.24, 24.06, 52.84, 457.29, 1845.21, 3781.73, 7596.18]
values_5_gpu = [16.81, 17.75, 19.50, 23.82, 28.88, 76.07, 135.25, 255.02]

values_10_cpu = [9.82, 21.87, 45.60, 90.66, 782.95, 3135.13, 6365.31, 12253]
values_10_gpu = [28.81, 30.80, 31.88, 35.48, 46.07, 135.50, 239.24, 404.57]

values_100_cpu = [87.01, 170.05, 348.09, 672.72, 5353.13, 21576.5]
values_100_gpu = [182.74, 211.10, 234.97, 247.05, 321.40, 1032.30]

sns.set(rc={'figure.figsize':(16,8)})

ax = sns.lineplot(x=sizes, y=values_5_cpu, label="CPU")
sns.lineplot(x=sizes, y=values_5_gpu, ax=ax, label = "GPU")
plt.title("Running times of K-means on CPU/GPU for different sizes of dataset with 5 groups")
plt.xlabel("Number of points")
plt.ylabel("Time in milliseconds")

plt.legend()
plt.savefig("statistics_5.png")
plt.clf()

sns.set(rc={'figure.figsize':(16,8)})

ax = sns.lineplot(x=sizes, y=values_10_cpu, label="CPU")
sns.lineplot(x=sizes, y=values_10_gpu, ax=ax, label = "GPU")
plt.title("Running times of K-means on CPU/GPU for different sizes of dataset with 10 groups")
plt.xlabel("Number of points")
plt.ylabel("Time in milliseconds")

plt.legend()
plt.savefig("statistics_10.png")
plt.clf()

sns.set(rc={'figure.figsize':(16,8)})

ax = sns.lineplot(x=sizes[:6], y=values_100_cpu, label="CPU")
sns.lineplot(x=sizes[:6], y=values_100_gpu, ax=ax, label = "GPU")
plt.title("Running times of K-means on CPU/GPU for different sizes of dataset with 100 groups")
plt.xlabel("Number of points")
plt.ylabel("Time in milliseconds")

plt.legend()
plt.savefig("statistics_100.png")
plt.clf()