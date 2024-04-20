import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import numpy as np
import math

class DraggablePoint:
    def __init__(self, point, ax, update_info_callback, color='blue', radius=0.1):
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
        self.circle = Circle(point, radius, facecolor=color, alpha=1, edgecolor='black')

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
    def __init__(self, source_points, target_points):
        self.source_points = source_points
        self.target_points = target_points
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.draggable_points = []
        self.init_plot()

    def update_info(self):
        coordinates = [dp.point for dp in self.draggable_points]
        coord_text = ', '.join([f'({x:.2f}, {y:.2f})' for x, y in coordinates])
        if hasattr(self, 'info_text'):
            self.info_text.set_text(f'Changed Coord: {coord_text}')
        else:
            # Adjust the text position to be within the visible area of the figure
            self.info_text = self.ax.figure.text(0.5, 1.05 , f'Points Coord: {coord_text}',
                                                 ha='center', va='bottom', transform=self.ax.transAxes, fontsize=10)
        self.fig.canvas.draw_idle()

    def init_plot(self):
        # plot setting
        self.ax.scatter(*zip(*self.target_points), color='red')
        for point in self.source_points:
            dp = DraggablePoint(point, self.ax, self.update_info, color='blue')
            self.draggable_points.append(dp)

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        self.update_info()

        # Add button for ICP calculation
        button_ax = self.fig.add_axes([0.35, 0.01, 0.2, 0.05])  
        self.button = Button(button_ax, 'Run ICP', color='lightblue', hovercolor='0.8')
        self.button.on_clicked(self.run_icp)


        plt.show()
        
    def run_icp(self, event):
        source_points_updated = [dp.point for dp in self.draggable_points]
        self.calculator = PointSetRegistrationCalculator(source_points_updated, self.target_points)
        
        initial_distance = self.calculator.calculate_average_distance()
        print(f"Initial average distance: {initial_distance}")

        updated_points = self.calculator.apply_icp()

        # 更新 draggable_points 为 ICP 后的位置，并重新绘制它们
        for dp, new_point in zip(self.draggable_points, updated_points):
            dp.point = new_point
            dp.circle.set_center(new_point)

        self.fig.canvas.draw_idle()

        final_distance = self.calculator.calculate_average_distance()
        print(f"Final average distance after ICP: {final_distance}")




# please check this class. Jack.
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

    def apply_icp(self, iterations=100):
        self.source_points = ICP(self.source_points, self.target_points)
        return self.source_points



def GetTheAngleOfTwoPoints(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

def GetDistOfTwoPoints(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def GetClosestID(p_x, p_y, pt_set):
    distances = [GetDistOfTwoPoints(p_x, p_y, pt[0], pt[1]) for pt in pt_set.T]
    return np.argmin(distances)

def DistOfTwoSet(set1, set2):
    total_distance = 0
    for i in range(set1.shape[1]):
        closest_id = GetClosestID(set1[0, i], set1[1, i], set2)
        total_distance += GetDistOfTwoPoints(set1[0, i], set1[1, i], set2[0, closest_id], set2[1, closest_id])
    return total_distance / set1.shape[1]

def ICP(sourcePoints, targetPoints):
    A = np.array(targetPoints)  # A is the target point cloud
    B = np.array(sourcePoints)  # B is the source point cloud

    iteration_times = 0
    dist_before = DistOfTwoSet(A, B)
    dist_improve = float('inf')

    while iteration_times < 10 and dist_improve > 0.001:
        # Compute centroids
        x_mean_target = np.mean(A[:, 0])
        y_mean_target = np.mean(A[:, 1])
        x_mean_source = np.mean(B[:, 0])
        y_mean_source = np.mean(B[:, 1])

        # Center the point clouds
        A_ = A - np.array([x_mean_target, y_mean_target])
        B_ = B - np.array([x_mean_source, y_mean_source])

        w_up = 0
        w_down = 0
        for i in range(B.shape[0]):  # 应该遍历 B 的每个点
            closest_id = GetClosestID(B_[i, 0], B_[i, 1], A_)
            w_up += B_[i, 0] * A_[closest_id, 1] - B_[i, 1] * A_[closest_id, 0]
            w_down += B_[i, 0] * A_[closest_id, 0] + B_[i, 1] * A_[closest_id, 1]

        # Compute the rotation angle and translation
        TheRadian = math.atan2(w_up, w_down)
        R = np.array([[math.cos(TheRadian), -math.sin(TheRadian)],
                      [math.sin(TheRadian), math.cos(TheRadian)]])
        t = np.array([x_mean_target, y_mean_target]) - R.dot(np.array([x_mean_source, y_mean_source]))

        # Apply the transformation to B
        B = (R.dot(B_.T)).T + t

        # Compute the new distance and improvement
        dist_now = DistOfTwoSet(A, B)
        dist_improve = dist_before - dist_now
        print(f"Iteration {iteration_times + 1}: Loss {dist_now}, Improvement {dist_improve}")

        dist_before = dist_now
        iteration_times += 1

    return B

 
# main function
if __name__ == '__main__':
    # Example usage
    # translation
    source_points = [(2, 1), (3, 4), (1, 3), (4, 3), (2, 2)]
    target_points = [(3, 2), (4, 5), (2, 4), (5, 4), (3, 3)]
    #target_points = [(7, 6), (8, 9), (6, 8), (9, 8), (7, 7)]

    # another example rotation
    # source_points = [(3.00, 2.00), (2.50, 1.00), (2.00, 2.50), (1.00, 2.00), (1.50, 1.00)]
    # target_points = [(6.5, 7.0), (7.0, 8.0), (7.5, 6.5), (8.5, 7.0), (8.0, 8.0)]

    # init the graph
    PointSetRegistration(source_points, target_points)

    # set up our calculator and ICP
    psr_calculator = PointSetRegistrationCalculator(source_points, target_points)
   
    initial_distance = psr_calculator.calculate_average_distance()
    print(f"Initial average distance: {initial_distance}")

    final_distance = psr_calculator.apply_icp()
    print(f"Final average distance after ICP: {final_distance}")