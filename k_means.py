import tkinter as tk
from tkinter import ttk


# Function to run the K-Means visualization
def run_kmeans(n_samples, centers, cluster_std):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    spread_centers = np.linspace(100, 100, centers)
    custom_centers = np.array(
        [[x, y] for x in spread_centers for y in spread_centers][:centers]
    )

    # Generate a synthetic dataset based on input values
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=custom_centers,
        cluster_std=cluster_std,
        random_state=10,
    )

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        num_points = min(frame, X.shape[0])
        if num_points > 1:
            kmeans = KMeans(n_clusters=centers, random_state=10).fit(X[:num_points])
            y_kmeans = kmeans.predict(X[:num_points])
            centers_loc = kmeans.cluster_centers_

            ax.scatter(X[:num_points, 0], X[:num_points, 1], c=y_kmeans, cmap="viridis")
            ax.scatter(centers_loc[:, 0], centers_loc[:, 1], c="red", s=200, alpha=0.5)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("K-Means Clustering Over Time")

    ani = FuncAnimation(
        fig, update, frames=np.arange(1, X.shape[0] + 20, 5), interval=100, repeat=False
    )
    plt.show()


# Function called when "Run" button is pressed
def on_run():
    n_samples = int(entry_n_samples.get())
    centers = int(entry_centers.get())
    cluster_std = float(entry_cluster_std.get())
    run_kmeans(n_samples, centers, cluster_std)


# Create the main window
root = tk.Tk()
root.title("K-Means Parameters")

# Create and grid the layout frames
frame_inputs = ttk.Frame(root)
frame_inputs.grid(row=0, column=0, sticky="ew")

# Labels and entries for parameters
ttk.Label(frame_inputs, text="Number of Samples:").grid(row=0, column=0, sticky="w")
entry_n_samples = ttk.Entry(frame_inputs)
entry_n_samples.grid(row=0, column=1, sticky="ew")

ttk.Label(frame_inputs, text="Centers:").grid(row=1, column=0, sticky="w")
entry_centers = ttk.Entry(frame_inputs)
entry_centers.grid(row=1, column=1, sticky="ew")

ttk.Label(frame_inputs, text="Cluster Std Dev:").grid(row=2, column=0, sticky="w")
entry_cluster_std = ttk.Entry(frame_inputs)
entry_cluster_std.grid(row=2, column=1, sticky="ew")

# Run button
button_run = ttk.Button(root, text="Run", command=on_run)
button_run.grid(row=1, column=0, sticky="ew")

root.mainloop()
