import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, TextBox
import numpy as np

import image_processing
import matplotlib.patches as patches
import random
import time


class DraggablePoint:
    """
    A class for draggable points, used to create and manage draggable points on a matplotlib plot.

    Parameters
    ----------
    point : tuple
        The initial coordinates (x, y) of the point.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    update_info_callback : function
        The callback function called when the point is moved.
    color : str
        The color of the point.
    radius : float
        The radius of the point.
    """

    def __init__(self, point, ax, update_info_callback, color='blue', radius=0.3):
        self.point = point
        self.ax = ax
        self.radius = radius
        self.color = color
        self.circle = Circle(point, radius, facecolor=color, alpha=1, edgecolor='black')
        self.ax.add_patch(self.circle)
        self.press = None
        self.update_info_callback = update_info_callback
        self.connect()

    def connect(self):
        """Connect all the matplotlib events."""
        self.cidpress = self.circle.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.circle.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidhover = self.circle.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)

    def on_hover(self, event):
        """Handle mouse hover events."""
        contains, _ = self.circle.contains(event)
        if contains:
            self.circle.set_facecolor('pink')
        else:
            self.circle.set_facecolor(self.color)
        self.circle.figure.canvas.draw_idle()

    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.circle.axes: return
        contains, attr = self.circle.contains(event)
        if not contains: return
        self.press = self.point[0], self.point[1], event.xdata, event.ydata

    def on_motion(self, event):
        """Handle mouse drag events."""
        if self.press is None: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point = (x0 + dx, y0 + dy)
        self.circle.center = self.point
        self.circle.figure.canvas.draw_idle()

    def on_release(self, event):
        """Handle mouse release events."""
        self.press = None
        self.update_info_callback()
        self.circle.figure.canvas.draw()


    # def disconnect(self):
    #     self.circle.figure.canvas.mpl_disconnect(self.cidpress)
    #     self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
    #     self.circle.figure.canvas.mpl_disconnect(self.cidmotion)
    #     self.circle.figure.canvas.mpl_disconnect(self.cidhover)


class PointSetRegistration:
    """
    A class for point set registration, used to demonstrate the ICP algorithm with an interactive interface.

    Parameters
    ----------
    source_points : list of tuples
        The source point set.
    target_points : list of tuples
        The target point set.
    """
    def __init__(self, source_points, target_points, transformation):
        self.source_points = source_points
        self.target_points = target_points
        self.fig, self.ax = plt.subplots(figsize=(10, 12))
        self.original_matrix = transformation
        self.prev_matrix = np.eye(3,3)
        # self.matrix_history = []
        self.iteration_to_display = 0

        self.state = 0
        self.init_start_screen()

    def init_start_screen(self):
        """Initialize the start screen, displaying the algorithm introduction and a start button."""
        self.ax.clear()
        self.ax.axis('off') 
        self.ax.text(0.5, 0.6, 'Welcome to the ICP Algorithm Demo\n', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=self.ax.transAxes, fontsize=14)
        self.ax.text(0.5, 0.5, 
                    'Key Concepts: \n\n'
                    '- \\textbf{Point Set Registration}: Techniques for finding the best alignment between point sets.\\\n'
                    '- \\textbf{Point Cloud}: A collection of points. cloud representing a 3D shape in space.\\\n'
                    '- \\textbf{Rigid Transformation}: A transformation that maintains distances and angles, such as rotation and translation.\\\n'
                    '- \\textbf{Matrix Transformation}: Applying linear algebraic transformations using matrices.\\\n'
                    '- \\textbf{Iterative Closest Point (ICP) Algorithm}: Minimizes differences between point sets by iteratively updating transformations.\\\n'
                    '- \\textbf{Error Metrics}: Measures to quantify alignment quality, typically involving distances between points after transformation.',
                    horizontalalignment='center', verticalalignment='center', 
                    transform=self.ax.transAxes, fontsize=12, usetex=True)
        self.start_button_ax = self.fig.add_axes([0.45, 0.2, 0.1, 0.05])  # Modify here, set it as a class attribute
        self.start_button = Button(self.start_button_ax, 'Continue', color='lightblue', hovercolor='0.8')
        self.start_button.on_clicked(self.next_stage)
        plt.show()

    def next_stage(self, event):
        """Handle the step-by-step display logic of the interface."""
        if self.state == 0:
            self.state = 1
            self.ax.clear()
            self.ax.axis('off') 
            self.ax.text(0.5, 0.5, 
                        '\\textbf{1. Normalize Point Clouds:} Normalize to zero mean.\n'
                        '   - Subtract the mean of each coordinate from all points.\n\n'
                        '\\textbf{2. Find Corresponding Points:} Identify nearest points in normalized clouds.\n'
                        '   - For each point in source, find closest point in target using Euclidean distance.\n\n'
                        '\\textbf{3. Compute Transformation Matrix:} Use SVD on corresponding pairs.\n'
                        '   - Decompose covariance matrix from point pairs to extract rotation and scaling.\n\n'
                        '\\textbf{4. Transform Source Points:} Apply computed rotation and translation.\n'
                        '   - Update source points by applying the transformation matrix.\n\n'
                        '\\textbf{5. Calculate Current Loss:} Measure alignment quality to decide further iterations.\n'
                        '   - Calculate mean distance between transformed source points and target.\n\n'
                        '\\textbf{6. Decide on Iteration Continuation:} Check if additional iterations are needed.\n'
                        '   - Continue if loss is above threshold or max iterations not reached.\n',
                        horizontalalignment='center', verticalalignment='center', 
                        transform=self.ax.transAxes, fontsize=14, usetex=True)
            self.fig.canvas.draw_idle()

        elif self.state == 1:
            self.start_button_ax.remove()  # Remove the axis containing the button
            self.fig.canvas.draw_idle()    # Redraw the canvas to reflect changes
            self.init_main_screen()

    def init_main_screen(self):
        """Initialize the main screen, displaying the coordinate plot and interface elements."""
        self.ax.clear()
        self.ax.axis('on') 
        self.fig.canvas.draw_idle()
        self.start_button.set_active(False)  # Disable and hide the start button
        self.draggable_points = []
        self.ax.scatter(*zip(*self.target_points), color='red')
        for point in self.source_points:
            dp = DraggablePoint(point, self.ax, self.update_info, color='blue')
            self.draggable_points.append(dp)

        # self.ax.set_facecolor('xkcd:black')
        # self.fig.patch.set_facecolor('xkcd:black')

        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.update_info()

        button_ax = self.fig.add_axes([0.75, 0.005, 0.1, 0.05])
        # button_ax = self.fig.add_axes([0.35, 0.01, 0.2, 0.05])  
        self.button = Button(button_ax, 'Run ICP', color='lightblue', hovercolor='0.8')
        self.button.on_clicked(self.run_icp)

        # sample matrix
        self.update_displayed_matrices(np.eye(3,3))

        plt.show()

    def update_source_points(self, new_source_points):
        self.source_points = np.array(new_source_points)

    def update_info(self):
        """Update the text information of the coordinates."""
        coordinates = [dp.point for dp in self.draggable_points]
        if hasattr(self, 'calculator'):
            self.calculator.update_source_points(coordinates)
        self.fig.canvas.draw_idle()

    def update_displayed_matrices(self, most_recent_matrix):
        # calculate cumulative ICP matrix gathered by previous iterations
        cumulative_matrix = most_recent_matrix @ self.prev_matrix
        self.prev_matrix = np.copy(cumulative_matrix)

        self.props = dict(boxstyle='round', facecolor='wheat', alpha=1) # I can't figure out how to remove a box from the plot once it's there. This just puts the new on top of the old
        self.textstr = f'Actual:\n{format_array_or_string(self.original_matrix)}\nICP Guess:\n{format_array_or_string(cumulative_matrix)}\nIteration Count:\n{self.iteration_to_display}'

        self.ax.text(0.05, 0.95, self.textstr, transform=self.ax.transAxes, fontsize=14, verticalalignment='top', bbox=self.props)

    def run_icp(self, event):
        """Execute one iteration of the ICP algorithm and update the interface."""
        self.update_info()
        
        if not hasattr(self, 'calculator'):
            self.calculator = PointSetRegistrationCalculator(self.source_points, self.target_points)

        # Perform one iteration of ICP and get the transformation and loss
        self.source_points, applied_matrix, self.iteration_to_display = self.calculator.apply_icp()
        current_loss = self.calculator.calculate_average_distance()

        # Update the display of source points
        for i, dp in enumerate(self.draggable_points):
            new_position = self.source_points[i]
            dp.point = new_position
            dp.circle.center = new_position
            dp.circle.figure.canvas.draw_idle()

        self.update_displayed_matrices(applied_matrix)

        self.fig.canvas.draw_idle()

class PointSetRegistrationCalculator:
    """
    A class for point set registration calculation, using the ICP algorithm to compute the optimal alignment of point sets.
    """
    def __init__(self, source_points, target_points):
        self.source_points = np.array(source_points)
        self.target_points = np.array(target_points)
        self.iteration = 0

    def update_source_points(self, new_source_points):
        """Update the source points with the new positions."""
        self.source_points = np.array(new_source_points)

    def calculate_average_distance(self):
        """Calculate the current average distance between aligned points."""
        distances = np.linalg.norm(self.source_points - self.target_points, axis=1)
        return np.mean(distances)

    def apply_icp(self):
        """
        Perform one iteration of the ICP algorithm:
        1) Normalize point clouds to have zero mean.
        2) Find corresponding points based on normalized point clouds.
        3) Compute the transformation matrix using SVD based on corresponding point pairs.
        4) Transform the source points using the computed matrix.
        5) Optionally, calculate and check the current loss for convergence.
        """

        # create the matrix to use to shift all the points. 
        # should correspond first to a rotation, then a translation
        whole_transformation = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])

        # Step 1: Normalize point clouds to have zero mean
        source_mean = np.mean(self.source_points, axis=0)
        target_mean = np.mean(self.target_points, axis=0)
        source_centered = self.source_points - source_mean
        target_centered = self.target_points - target_mean

        # Step 2: Find corresponding points based on normalized point clouds
        closest_points = np.array([target_centered[self.get_closest_id(x, y, target_centered)] for x, y in source_centered])

        # Step 3: Compute the covariance matrix and apply SVD
        H = source_centered.T @ closest_points
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Step 4: Compute translation
        t = target_mean - R @ source_mean

        # transformed_points = np.dot(self.source_points, R.T) + t Alternative method of calculation. Not as clean

        # Transform the original source points using the computed rotation and translation
        whole_transformation[0,2] = t[0]
        whole_transformation[1,2] = t[1]

        whole_transformation[0,0] = R[0,0]
        whole_transformation[0,1] = R[0,1]
        whole_transformation[1,0] = R[1,0]
        whole_transformation[1,1] = R[1,1]

        transformed_points = del_extra_column((whole_transformation @ add_extra_column(self.source_points).T).T)

        self.source_points = transformed_points
        self.iteration += 1

        return self.source_points, whole_transformation, self.iteration

    def get_closest_id(self, x, y, target_centered):
        """Find the index of the closest point in normalized target_points."""
        distances = np.linalg.norm(target_centered - np.array([x, y]), axis=1)
        return np.argmin(distances)

def center_points_around_origin(points):
    centroid = points.mean(axis=0)

    return points - centroid 

def slice_up_raw_points(raw_points, one_over_resolution=100):

    source_points = np.asarray(raw_points.copy()[::one_over_resolution])
    target_points = raw_points.copy().tolist()

    offset_points = []

    for _ in range(int(one_over_resolution / 2)):
        offset_points.append(target_points.pop(0))
    for _ in range(int(one_over_resolution / 2)):
        target_points.append(offset_points.pop(0))

    target_points = np.asarray(target_points[::one_over_resolution])
    target_points = source_points # remove this if we want the points to be different. This is just testing

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

def example1(): # simple translation
    # translation
    source_points = [(2, 1), (3, 4), (1, 3), (4, 3), (2, 2)]
    target_points = [(7, 6), (8, 9), (6, 8), (9, 8), (7, 7)]
    transformation_matrix = [[0]]

    # rotation
    # source_points = [(3.00, 2.00), (2.50, 1.00), (2.00, 2.50), (1.00, 2.00), (1.50, 1.00)]
    # target_points = [(6.5, 7.0), (7.0, 8.0), (7.5, 6.5), (8.5, 7.0), (8.0, 8.0)]

    psr = PointSetRegistration(source_points, target_points, transformation_matrix)

    # set up our calculator and ICP
    psr_calculator = PointSetRegistrationCalculator(source_points, target_points)

def example2(): # simple rotation

    # rotation
    source_points = [(3.00, 2.00), (2.50, 1.00), (2.00, 2.50), (1.00, 2.00), (1.50, 1.00)]
    target_points = [(6.5, 7.0), (7.0, 8.0), (7.5, 6.5), (8.5, 7.0), (8.0, 8.0)]
    transformation_matrix = [[0]]

    psr = PointSetRegistration(source_points, target_points, transformation_matrix)

    # set up our calculator and ICP
    psr_calculator = PointSetRegistrationCalculator(source_points, target_points)

def example3(): # logo translation
    raw_points = np.asarray(image_processing.sample_data("images/logo.png"))
    centered_points = center_points_around_origin(raw_points)
    source_points, target_points = slice_up_raw_points(centered_points, 1000)

    ty      = -10
    tx      = -2
    theta   = 0

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

    # init the graph
    psr = PointSetRegistration(source_points, target_points, transformation_matrix)

    # set up our calculator and ICP
    psr_calculator = PointSetRegistrationCalculator(source_points, target_points)
    # initial_distance = psr_calculator.calculate_average_distance()

def example4(): # logo translation and slight rotation
    raw_points = np.asarray(image_processing.sample_data("images/logo.png"))
    centered_points = center_points_around_origin(raw_points)
    source_points, target_points = slice_up_raw_points(centered_points, 1000)

    ty      = -10
    tx      = -2
    theta   = np.pi/10

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

    # init the graph
    psr = PointSetRegistration(source_points, target_points, transformation_matrix)

    # set up our calculator and ICP
    psr_calculator = PointSetRegistrationCalculator(source_points, target_points)
    # initial_distance = psr_calculator.calculate_average_distance()

# main function
if __name__ == '__main__':
    example4()
