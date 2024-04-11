import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from sklearn.covariance import EllipticEnvelope

# Generate a synthetic dataset
X, _ = make_blobs(n_samples=3000, centers=[[50, 50]], cluster_std=5)

# Initialize the figure and axis
fig, ax = plt.subplots()
ax.set_title("Elliptic Envelope Outlier Detection")
ax.set_xlabel("Users Logging in on the Weekend")
ax.set_ylabel("Users Logging in on the Week")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)


# Function to update the plot with each frame
def update(frame):
    ax.clear()

    # Setting the title and labels again since ax.clear() clears everything
    ax.set_title("Elliptic Envelope Outlier Detection")
    ax.set_xlabel("Users Logging in on the Weekend")
    ax.set_ylabel("Users Logging in on the Week")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Apply Elliptic Envelope to the current subset of data
    current_data = X[:frame]
    outlier_detect = EllipticEnvelope(contamination=0.001)
    outlier_detect.fit(current_data)
    predictions = outlier_detect.predict(current_data)

    # Plotting current data points
    ax.scatter(
        current_data[:, 0],
        current_data[:, 1],
        c=predictions,
        cmap="Paired",
        marker="o",
        edgecolor="k",
    )

    # Calculate and plot decision boundary
    xx, yy = np.meshgrid(
        np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500),
        np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 500),
    )
    Z = outlier_detect.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="r", linestyles="dashed")


# Creating animation
ani = FuncAnimation(fig, update, frames=range(2, len(X) + 1, 10), repeat=False)
plt.show()
