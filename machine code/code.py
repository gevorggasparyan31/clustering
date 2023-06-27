import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# նկարների բեռնում
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')
image3 = cv2.imread('image3.jpg')
images = np.array([image1, image2, image3])

# նկարների մասշտաբավորում
max_width = max([image.shape[1] for image in images])
max_height = max([image.shape[0] for image in images])
resized_images = [cv2.resize(image, (max_width, max_height)) for image in images]

# նկարների բնութագրերի ստացում
features = np.vstack([image.reshape(-1, 3) for image in resized_images])

# K-միջինների մեթոդով կլաստերիզացիա
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features)
labels = kmeans.labels_.astype(int)

# կլաստերիզացման արդյունքների գրաֆիկական տեսք
colors = ['#FF5733', '#FFC300', '#DAF7A6'] # կլաստերների գունավորում
plt.scatter(features[:, 0], features[:, 1], c=[colors[label] for label in labels], s=1) # scatter plot
plt.axis('off')
plt.show()

# արտատպում
for i in range(n_clusters):
    indices = (labels == i).nonzero()[0]
    if len(indices) > 0:
        cluster = [resized_images[j] for j in indices if j < len(resized_images)]
        plt.figure(figsize=(10, 10))
        for j, image in enumerate(cluster):
            plt.subplot(1, len(cluster), j+1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.show()
