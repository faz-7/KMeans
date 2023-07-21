import cv2
import numpy as np
import matplotlib.pyplot as plt


# implementation of KMeans Algorithm
class KMeans:
    def __init__(self, n_clusters, max_iter=300, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        # select random center
        rng = np.random.RandomState(self.random_state)
        i = rng.permutation(X.shape[0])[:self.n_clusters]
        self.centers = X[i]

        # calculate distance of each point to the centers
        for _ in range(self.max_iter):
            distances = np.sqrt(((X - self.centers[:, np.newaxis]) ** 2).sum(axis=2))

            # assign each point to the nearest center
            labels = np.argmin(distances, axis=0)

            # update center of clusters
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # check changes in cluster centers
            if np.allclose(self.centers, new_centers):
                break

            self.centers = new_centers

        self.labels_ = labels
        self.cluster_centers_ = self.centers

    def predict(self, X):
        distances = np.sqrt(((X - self.centers[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels


def convert_image2matrix(path):
    img = cv2.imread(path)  # read image from path
    img = img[:, :, 0]  # extract the first color channel
    digit_size = 16  # set the size of each digit image

    # initialize an empty list to store digit images
    digit_images = []

    # loop through each row and column of the image
    for i in range(1, 35):
        for j in range(1, 34):
            # extract the digit image from the image
            digit_img = img[(i - 1) * digit_size:i * digit_size, (j - 1) * digit_size:j * digit_size]

            # reshape the digit image into a 1D array
            x = np.reshape(digit_img, [1, digit_size * digit_size])

            # append the digit image array to the list
            if not (i > 12 and j == 33):
                digit_images.append(x)

    arr = np.vstack(digit_images)  # stack all the digit images into a single 2D array
    return arr


img = convert_image2matrix('usps_5.jpg')
data = img.reshape(-1, 256)  # Reshape the image into a 1100x256 matrix

# Perform K-means clustering with k=10
k = 7
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(data)
centers = kmeans.cluster_centers_

# Display the 16x16 images corresponding to each cluster center
plt.figure(figsize=(k, 5))
for i in range(k):
    plt.subplot(2, k, i + k + 1)
    plt.imshow(np.reshape(centers[i, :], (16, 16)), cmap='gray')

plt.show()