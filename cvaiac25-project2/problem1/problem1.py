import os
from load_data import load_data

DATA_FILENAME = 'demo.p'  # TODO: Change to data.p for your final submission


def task_1(data):
    """
    Task 1: Given the intrinsic and extrinsic projection matrices, project the point cloud onto the image of Cam 2. Color each point projected according to their respective semantic label. The color map for each label is provided with the data.
    """
    pass

def task_2(data):
    """
    Task 2: In addition to the points with semantic labels, project the 3D bounding boxes of all given vehicles onto the Cam 2 image.
    """
    pass


if __name__ == "__main__":
    data_path = os.path.join('data', DATA_FILENAME)
    data = load_data(data_path)

    task_1(data)
    task_2(data)