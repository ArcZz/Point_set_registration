import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, TextBox
import numpy as np
import math

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
    def __init__(self, point, ax, update_info_callback, color='blue', radius=0.1):
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

    def disconnect(self):
        """Disconnect all matplotlib event connections."""
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)
        self.circle.figure.canvas.mpl_disconnect(self.cidhover)

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
    def __init__(self, source_points, target_points):
        self.source_points = source_points
        self.target_points = target_points
        self.fig, self.ax = plt.subplots(figsize=(10, 12))
        self.state = 0
        self.init_start_screen()

    def init_start_screen(self):
        """Initialize the start screen, displaying the algorithm introduction and a start button."""
        self.ax.clear()
        self.ax.axis('off') 
        self.ax.text(0.5, 0.6, 'Welcome to the ICP Algorithm Demo', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=self.ax.transAxes, fontsize=14)
        self.ax.text(0.5, 0.5, 'Key Concepts:\n- \\textbf{Point Set Registration}: Techniques for finding the best alignment between point sets.\n- \\textbf{Point Cloud}: A collection of points in space representing a 3D shape.\n- \\textbf{Rigid Transformation}: A transformation that maintains distances and angles, such as rotation and translation.\n- \\textbf{Matrix Transformation}: Applying linear algebraic transformations using matrices.',
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
            self.ax.text(0.5, 0.5, 'ICP Algorithm Steps:\n\n1. Select corresponding points between datasets.\n2. Estimate the transformation that best aligns the points.\n3. Apply the transformation to align the point sets.\n4. Iterate until convergence.\n5. Evaluate the quality of the alignment.',
                        horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=14)
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
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 12)
        self.ax.set_aspect('equal')
        self.update_info()
        button_ax = self.fig.add_axes([0.75, 0.005, 0.1, 0.05])
        self.button = Button(button_ax, 'Run ICP', color='lightblue', hovercolor='0.8')
        self.button.on_clicked(self.run_icp)
        plt.show()

    def update_info(self):
        """Update the text information of the coordinates."""
        coordinates = [dp.point for dp in self.draggable_points]
        coord_text = ', '.join([f'({x:.2f}, {y:.2f})' for x, y in coordinates])
        if hasattr(self, 'info_text'):
            self.info_text.set_text(f'Changed Coord: {coord_text}')
        else:
            self.info_text = self.ax.figure.text(0.5, 1.05, f'Points Coord: {coord_text}',
                                                 ha='center', va='bottom', transform=self.ax.transAxes, fontsize=10)
        self.fig.canvas.draw_idle()


    def run_icp(self, event):
        """Execute one iteration of the ICP algorithm and update the interface."""
        if not hasattr(self, 'calculator'):
            self.calculator = PointSetRegistrationCalculator(self.source_points, self.target_points)

        # Perform one iteration of ICP and get the transformation and loss
        self.source_points, R, t = self.calculator.apply_icp_iteration()
        current_loss = self.calculator.calculate_average_distance()

        # Update the display of source points
        for i, dp in enumerate(self.draggable_points):
            new_position = self.source_points[i]
            dp.point = new_position
            dp.circle.center = new_position
            dp.circle.figure.canvas.draw_idle()

        # Update the matrix textbox with current iteration info
        matrix_text = f'Iteration #{self.calculator.iteration}\nRotation:\n{R}\nTranslation:\n{t}\nCurrent Loss: {current_loss:.4f}'
        if hasattr(self, 'matrix_textbox'):
            self.matrix_textbox.set_val(matrix_text)
        else:
            matrix_box_ax = self.fig.add_axes([0.05, 0.01, 0.25, 0.1])
            self.matrix_textbox = TextBox(matrix_box_ax, '', initial=matrix_text)
            self.matrix_textbox.cursor.set_color('none')
            # Override the on_submit method to prevent user typing
            self.matrix_textbox.on_submit = lambda x: None


        self.fig.canvas.draw_idle()


class PointSetRegistrationCalculator:
    """
    A class for point set registration calculation, used to compute the minimum distance between point sets and perform the ICP algorithm.

    Parameters
    ----------
    source_points : list of tuples
        The source point set.
    target_points : list of tuples
        The target point set.
    """
    def __init__(self, source_points, target_points):
        self.source_points = np.array(source_points)
        self.target_points = np.array(target_points)
        self.iteration = 0
        self.R = None
        self.t = None

    def calculate_average_distance(self):
        distances = np.sqrt(((self.source_points - self.target_points) ** 2).sum(axis=1))
        return np.mean(distances)

    def apply_icp_iteration(self):
        # Perform one iteration of ICP
        if self.iteration == 0:
            self.R = np.eye(2)  # Initial rotation matrix
            self.t = np.zeros(2)  # Initial translation vector

        closest_points = np.array([
            self.target_points[self.get_closest_id(x, y)]
            for x, y in self.source_points
        ])

        # Compute centroids of source and target
        source_centroid = np.mean(self.source_points, axis=0)
        target_centroid = np.mean(closest_points, axis=0)

        # Subtract centroids
        source_centered = self.source_points - source_centroid
        target_centered = closest_points - target_centroid

        # Compute the matrix W
        W = np.zeros((2, 2))
        for s, t in zip(source_centered, target_centered):
            W += np.outer(s, t)

        # Compute rotation using arctangent
        theta = math.atan2(W[1, 0] - W[0, 1], W[0, 0] + W[1, 1])
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        
        # Compute translation
        t = target_centroid - np.dot(R, source_centroid)

        # Accumulate rotation and translation
        self.R = np.dot(R, self.R)
        self.t = t + np.dot(R, self.t)

        # Apply transformation
        self.source_points = (np.dot(self.source_points, self.R.T) + self.t).tolist()

        # Increment the iteration counter
        self.iteration += 1

        return self.source_points, self.R, self.t

    def get_closest_id(self, x, y):
        # Find the index of the closest point in target_points
        distances = np.linalg.norm(self.target_points - np.array([x, y]), axis=1)
        return np.argmin(distances)
    
    
if __name__ == '__main__':
    source_points = [(2, 1), (3, 4), (1, 3), (4, 3), (2, 2)]
    target_points = [(3, 2), (4, 5), (2, 4), (5, 4), (3, 3)]
    PointSetRegistration(source_points, target_points)
