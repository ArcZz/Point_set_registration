import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import numpy as np
import image_processing
import matplotlib.patches as patches
import random
import time


class DraggablePoint:
    def __init__(self, point, ax, update_info_callback, color='blue', radius=.3):
        """
        Initialize a draggable point on a matplotlib axis.

        :param point: Initial coordinates of the point (x, y).
        :param ax: Matplotlib axis where the point will be drawn.
        :param update_info_callback: Function to call when the point is moved.
        :param color: Color of the point.
        :param radius: Radius of the point.
        """
        self.point = point
        self.ax = ax
        self.radius = radius
        self.color = color
        self.circle = Circle(point, radius, color=color, alpha=1)
        self.ax.add_patch(self.circle)
        self.press = None
        self.update_info_callback = update_info_callback
        self.connect()

    def connect(self):
        self.cidpress = self.circle.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.circle.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidhover = self.circle.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)
        

    def on_hover(self, event):
        contains, _ = self.circle.contains(event)
        if contains:
            self.circle.set_facecolor('pink')  

        else:
            self.circle.set_facecolor(self.color)
        self.circle.figure.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.circle.axes: return
        contains, attr = self.circle.contains(event)
        if not contains: return
        self.press = self.point[0], self.point[1], event.xdata, event.ydata

    def on_motion(self, event):
        if self.press is None: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point = (x0 + dx, y0 + dy)
        self.circle.center = self.point
        self.circle.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.update_info_callback()
        self.circle.figure.canvas.draw()

    def disconnect(self):
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)
        self.circle.figure.canvas.mpl_disconnect(self.cidhover)


class PointSetRegistration:
    """
    Initialize the point set registration with source and target points.
    """
    def __init__(self, source_points, target_points, transformation):
        self.source_points = source_points
        self.target_points = target_points
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.matrix = transformation

        self.draggable_points = []
        self.init_plot()

    def update_info(self):
        coordinates = [dp.point for dp in self.draggable_points]
        coord_text = ', '.join([f'({x:.2f}, {y:.2f})' for x, y in coordinates])
        if hasattr(self, 'info_text'):
            self.info_text.set_text(f'Changed Coord: {coord_text}')
        else:
            # Adjust the text position to be within the visible area of the figure
            self.info_text = self.ax.figure.text(0.5, 1.05 , 'Point Set Registration',
                                                 ha='center', va='bottom', fontsize=10)
        self.fig.canvas.draw_idle()

    def init_plot(self):
        # plot setting
        self.ax.scatter(*zip(*self.target_points), color='red')
        for point in self.source_points:
            dp = DraggablePoint(point, self.ax, self.update_info, color='blue')
            self.draggable_points.append(dp)

        self.ax.set_facecolor('xkcd:black')
        self.fig.patch.set_facecolor('xkcd:black')


        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.update_info()

        # Add button for ICP calculation
        button_ax = self.fig.add_axes([0.35, 0.01, 0.2, 0.05])  
        self.button = Button(button_ax, 'Run ICP', color='lightblue', hovercolor='0.8')
        self.button.on_clicked(self.run_icp)

        # sample matrix
        self.update_displayed_matrices("N/a")

        plt.show()


    def update_displayed_matrices(self, original):
        # these are matplotlib.patch.Patch properties
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=1) # I can't figure out how to remove a box from the plot once it's there. This just puts the new on top of the old
        self.textstr = f'Actual:\n{format_array_or_string(self.matrix)}\nICP Guess:\n{format_array_or_string(original)}'

        self.ax.text(0.05, 0.95, self.textstr, transform=self.ax.transAxes, fontsize=14, verticalalignment='top', bbox=self.props)

        
    def run_icp(self, event):
        calculator = PointSetRegistrationCalculator([dp.point for dp in self.draggable_points], self.target_points)
        initial_distance = calculator.calculate_average_distance()
        print(f"Initial average distance: {initial_distance}")

        final_distance, applied_matrix, final_source_points = calculator.apply_icp()
        self.source_points = final_source_points
        print(f"Final average distance after ICP: {final_distance}")

        for dp, new_point in zip(self.draggable_points, calculator.source_points):
            dp.point = new_point
            dp.circle.set_center(new_point)

        self.update_displayed_matrices(applied_matrix)

        self.fig.canvas.draw_idle()



class PointSetRegistrationCalculator:
    """
    After user clicks on the "Run ICP" button, calculate the average distance between source and target points.
    Apply the Iterative Closest Point (ICP) algorithm
    this not finished yet. jack you can check this method and discuss with me later. but this gonna be the main part of the code in future.


    :param iterations: Number of iterations to perform ICP.
    :return: The average distance after applying ICP.
    """
    def __init__(self, source_points, target_points):
        self.source_points = np.array(source_points)
        self.target_points = np.array(target_points)

    def calculate_average_distance(self):
        distances = np.sqrt(np.sum((self.source_points - self.target_points) ** 2, axis=1))
        return np.mean(distances)

    def apply_icp(self, iterations=1):
        for _ in range(iterations):
            # # Step 1: find centroid
            # centroid_source = np.mean(self.source_points, axis=0)
            # centroid_target = np.mean(self.target_points, axis=0)
            # source_centered = self.source_points - centroid_source
            # target_centered = self.target_points - centroid_target

            # # self.source_points = source_centered + centroid_target
            
            # H = source_centered.T @ target_centered
            # print(np.shape(H))
            # U, _, Vt = np.linalg.svd(H)
            # R = Vt.T @ U.T
            # if np.linalg.det(R) < 0:
            #     Vt[-1, :] *= -1
            #     R = Vt.T @ U.T
            # # t = centroid_target - R @ centroid_source
            # t = centroid_target

            # # R = np.eye(2,2)
            # # t = 0

            # self.source_points = (R @ source_centered.T).T + t

            #NEXT TODO
            # implement one where it explicitly looks for the closest target point for each source point


            # create the matrix to use to shift all the points. 
            # should correspond first to a rotation, then a translation
            whole_transformation = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])


            # Step 1: find centroid of target points and move source there
            source_centroid = np.mean(self.source_points, axis=0)
            target_centroid = np.mean(self.target_points, axis=0)

            # point sets from the reference frame of their center of mass
            source_centered = self.source_points - source_centroid
            target_centered = self.target_points - target_centroid
            
            # takes inner product of the two sets of centered points
            H = source_centered.T @ target_centered
            print(np.linalg.svd(H))

            # performs single value composition to determine what correspondences the points have
            U, _, Vt = np.linalg.svd(H)

            # extracts the rotation matrix from the singular values
            R = Vt.T @ U.T

            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            t = target_centroid 


            self.source_points = (R @ source_centered.T).T + t

            # offset = target_centroid - source_centroid
            # if abs(offset[0]) < 1e-10:
            #     offset[0] = 0
            # if abs(offset[1]) < 1e-10:
            #     offset[1] = 0
            
            # shift_centroid = np.append(offset, 0)

            # source_centroid_matrix = np.zeros((3,3))
            # source_centroid_matrix[0,2] = shift_centroid[0]
            # source_centroid_matrix[1,2] = shift_centroid[1]

            
            # whole_transformation = whole_transformation + source_centroid_matrix

            # # print(whole_transformation)
            # transformed_points = whole_transformation @ add_extra_column(self.source_points).T
            # self.source_points = del_extra_column(transformed_points.T)

        return self.calculate_average_distance(), whole_transformation, self.source_points


    # JHH TODOS
    # display the source points translated away from the originals (compare the matrices later??)
    # make a way to show matrices
    # show the original matrix, and the one that is updated every time an iteration runs.
        # they should converge, I guess? maybe make the color of each number in the matrix 
        # indicative of how close it is to the original displacement...


    def update_displayed_matrices():
        None
 


def center_points_around_origin(points):
    centroid = points.mean(axis=0)

    return points - centroid 


def slice_up_raw_points(raw_points):
    one_over_num_points = 100

    source_points = np.asarray(raw_points.copy()[::one_over_num_points])
    target_points = raw_points.copy().tolist()

    offset_points = []

    for _ in range(int(one_over_num_points / 2)):
        offset_points.append(target_points.pop(0))
    for _ in range(int(one_over_num_points / 2)):
        target_points.append(offset_points.pop(0))

    target_points = np.asarray(target_points[::one_over_num_points])
    target_points = source_points # remove this later. This is just testing

    return source_points, target_points

def add_extra_column(points):
    num_rows = np.shape(points)[0]
    row_to_add_for_translation = np.ones((num_rows,1))

    # Add extra column to the matrix
    return np.hstack((points, row_to_add_for_translation))

def del_extra_column(points):
    return np.delete(points, -1, axis=1)

def format_array_or_string(value):
    if isinstance(value, np.ndarray):
        return np.array2string(value, formatter={"float_kind": lambda x: f"{x:.2f}"})
    return str(value)


# main function
if __name__ == '__main__':
    raw_points = np.asarray(image_processing.sample_data())

    centered_points = center_points_around_origin(raw_points)


    source_points, target_points = slice_up_raw_points(centered_points)
    
    random.shuffle(source_points)
    random.shuffle(target_points)

    ty      = -2
    tx      = 5
    theta   = np.pi / 2

    transformation_matrix = np.array([
        [np.cos(theta), - np.sin(theta),    tx],
        [np.sin(theta), np.cos(theta),      ty],
        [0            , 0            ,      1],
    ])

    # Add extra column to target points so matrix works
    target_points = add_extra_column(target_points)

    # perform the linear transformation
    target_points = transformation_matrix @ target_points.T
    target_points = target_points.T # target points need to be column vectors

    target_points = del_extra_column(target_points)
    
    # Example usage
    # translation
    # source_points = [(2, 1), (3, 4), (1, 3), (4, 3), (2, 2)]
    # target_points = [(7, 6), (8, 9), (6, 8), (9, 8), (7, 7)]

    # another example rotation
    # source_points = [(3.00, 2.00), (2.50, 1.00), (2.00, 2.50), (1.00, 2.00), (1.50, 1.00)]
    # target_points = [(6.5, 7.0), (7.0, 8.0), (7.5, 6.5), (8.5, 7.0), (8.0, 8.0)]

    # init the graph
    psr = PointSetRegistration(source_points, target_points, transformation_matrix)

    # set up our calculator and ICP
    psr_calculator = PointSetRegistrationCalculator(source_points, target_points)
    initial_distance = psr_calculator.calculate_average_distance()


    # print(f"Initial average distance: {initial_distance}")

    # final_distance = psr_calculator.apply_icp()
    # print(f"Final average distance after ICP: {final_distance}")
