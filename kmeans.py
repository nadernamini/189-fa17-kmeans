import numpy as np


class Kmeans:
    def __init__(self, data, labels, k=25):
        self.partitions = [set() for _ in range(k)]
        self.means = []
        self.labeled_means = []
        self.k = k
        self.data = data
        self.label = labels
        # K-means++
        self.initialize_means()

    def initialize_means(self):
        """
        Initialize the means using the K-means++ algorithm
        :return: None
        """

        def find_nearests():
            nearests = []
            for i in range(self.data.shape[0]):
                dist, nearest = float('inf'), i
                for j, m in enumerate(means):
                    new = np.linalg.norm(m - self.data[i])
                    if new < dist:
                        dist = new
                        nearest = j
                nearests.append(nearest)
            return nearests

        def find_p(nearests):
            ps = []
            for i in range(self.data.shape[0]):
                ps.append(np.linalg.norm(means[nearests[i]] - self.data[i]) ** 2)
            sm = sum(ps)
            ps = [p / sm for p in ps]
            return ps

        means = []
        seen = set()
        p = [1 / np.shape(self.data)[0] for _ in range(np.shape(self.data)[0])]
        while len(means) < self.k:
            idx = np.random.choice(self.data.shape[0], 1, p=p, replace=False)[0]
            print(idx)
            if idx not in seen:
                seen.add(idx)
                mean = self.data[idx]
                means.append(mean)
                nearest_means = find_nearests()
                p = find_p(nearest_means)

        self.means = means

    def find_closest(self, i, data):
        """
        finds the index of the closest mean.
        :param i: the i-th row of the train_data to look at
        :param data: the data to use for finding the closest mean
        :return: the index of the closest mean
        """
        dist, k = float('inf'), i
        row = data[i]
        for j, mean in enumerate(self.means):
            new = np.linalg.norm(mean - row) ** 2
            if new < dist:
                dist, k = new, j
        return k

    def assignment_step(self):
        """
        Performs the assignment step in Lloyd's algorithm
        :return: whether or not the assignments were changed
        """
        changed = False
        for i in range(self.data.shape[0]):
            k = self.find_closest(i, self.data)
            if i not in self.partitions[k]:
                changed = True
                self.partitions[k].add(i)
        return changed

    def update_step(self):
        """
        Performs the update step in Lloyd's algorithm
        :return: None
        """
        for i in range(self.k):
            new_mean, size = np.zeros(self.data[i].shape), len(self.partitions[i])
            for j in self.partitions[i]:
                new_mean += self.data[j]
            new_mean = new_mean / size
            self.means[i] = new_mean

    def run(self):
        """
        run Lloyd's algorithm until convergence
        :return: None
        """
        changed = True
        while changed:
            changed = self.assignment_step()
            self.update_step()
        self.label_centroids()

    def label_centroids(self):
        """
        labels the centroids to have the name of the most common label
        :return: None
        """
        labels = []
        for i in range(self.k):
            unique_labels, counts = np.unique(self.label[list(self.partitions[i])], return_counts=True)
            label = unique_labels[np.argmax(counts)]
            labels.append(label)
        self.labeled_means = labels

    def classify(self, data):
        labels = np.zeros((data.shape[0],))
        for i in range(data.shape[0]):
            k = self.find_closest(i, data)
            labels[i] = self.labeled_means[k]
        return labels
