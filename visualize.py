import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from utils import boxes_to_corners_3d


def visual_right_scale(pos, ax):
    max_range = np.array([pos[:, 0].max()-pos[:, 0].min(),
                          pos[:, 1].max()-pos[:, 1].min(),
                          pos[:, 2].max()-pos[:, 2].min()]).max() / 2.0

    mid_x = (pos[:, 0].max()+pos[:, 0].min()) * 0.5
    mid_y = (pos[:, 1].max()+pos[:, 1].min()) * 0.5
    mid_z = (pos[:, 2].max()+pos[:, 2].min()) * 0.5

    # make scale look equal
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_3d_boxes(corners3d, ax):
    '''
    corners3d: (N, 8, 3)
    '''
    for n in range(corners3d.shape[0]):
        b = corners3d[n]  # (8, 3)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            ax.plot([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]],
                    color='r')

            i, j = k + 4, (k + 1) % 4 + 4
            ax.plot([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]],
                    color='r')

            i, j = k, k + 4
            ax.plot([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]],
                    color='r')
    return


def sample(pos, n):
    num = pos.shape[0]
    idx = np.random.choice(num, n, False)
    return pos[idx]


parser = argparse.ArgumentParser()
parser.add_argument("--i", type=int, default=0, help="points_{i}.npy")
parser.add_argument("--idx", type=str, default='000936',
                    help='specify data index: {idx}.bin')
parser.add_argument('--category', type=str, default='car',
                    help='specify the category' + 
                        '{ \
                            Car, \
                            Van, \
                            Truck, \
                            Pedestrian, \
                            Person_sitting, \
                            Cyclist, \
                            Tram \
                        }')
args = parser.parse_args()

######### Visualize in matplotlib ########
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# pts_path = 'output/points.npy'
# bbox_path = 'output/bboxes.npy'
pts_path = 'output/{}_{}_point_{}.npy'.format(args.idx, args.category, args.i)
bbox_path = 'output/{}_{}_bbox_{}.npy'.format(args.idx, args.category, args.i)
pts = np.load(pts_path).reshape(-1, 3)
bbox = np.load(bbox_path).reshape(-1, 7)
corners3d = boxes_to_corners_3d(bbox)
# pts = sample(pts, 5000)

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, c='g', lw=0, alpha=1)
visualize_3d_boxes(corners3d, ax)

visual_right_scale(corners3d.reshape(-1, 3), ax)
ax.title.set_text(args.category)
ax.view_init(elev=120., azim=-90)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()



