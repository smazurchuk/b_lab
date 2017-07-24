

# A plotting function for how I defined the error

import numpy as np
import matplotlib.pyplot as plt

def labeled_plot(Bdata, Gdata, labels, text, title, fname):
    # Calculate Centroids
    numL = len(set(labels))
    B_centroids = np.zeros((numL,2))
    B_cDist = np.zeros((numL,))
    for label in range(numL):
        B_centroids[label] = np.mean(Bdata[labels==label],axis=0)
    for idx, row in enumerate(Bdata):
        B_cDist[int(labels[idx])] = B_cDist[int(labels[idx])] + np.linalg.norm(row - B_centroids[int(labels[idx])])

    G_centroids = np.zeros((numL,2))
    G_cDist = np.zeros((numL,))
    for label in range(numL):
        G_centroids[label] = np.mean(Gdata[labels==label],axis=0)
    for idx, row in enumerate(Gdata):
        G_cDist[int(labels[idx])] = G_cDist[int(labels[idx])] + np.linalg.norm(row - G_centroids[int(labels[idx])])

    plt.figure(1, figsize=(32,12))
    plt.subplot(121)
    plt.scatter(Bdata[:,0], Bdata[:,1], s=5, c=labels)
    plt.scatter(B_centroids[:,0],B_centroids[:,1], s=80, c=list(range(numL)))
    for i in labels:
        plt.annotate(text[int(i)], (B_centroids[int(i),0],B_centroids[int(i),1]))
    plt.title('Binder tSNE Reduced', fontweight='bold')

    plt.subplot(122)
    plt.scatter(Gdata[:,0], Gdata[:,1], s=5, c=labels)
    plt.scatter(G_centroids[:,0],G_centroids[:,1], s=80, c=list(range(numL)))
    for i in labels:
        plt.annotate(text[int(i)], (G_centroids[int(i),0],G_centroids[int(i),1]))
    plt.title('Google tSNE Reduced', fontweight='bold')
    plt.suptitle(title, fontweight='bold')
    figD = plt.gcf()
    plt.show()
    figD.savefig(fname, dpi=600)
