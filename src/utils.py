import matplotlib.pyplot as plt
import seaborn as sns

def plot_elbow(k_values, inertia_values):
    plt.figure()
    plt.plot(k_values, inertia_values, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()


def plot_clusters(pca_df):
    plt.figure()
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster",
        data=pca_df,
        palette="viridis"
    )
    plt.title("Customer Segments (PCA View)")
    plt.show()