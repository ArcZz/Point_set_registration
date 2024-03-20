import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import numpy as np

class DraggablePoint:
    def __init__(self, point, ax, update_info_callback, color='blue', radius=0.1):
        self.point = point
        self.ax = ax
        self.radius = radius
        self.color = color
        self.circle = Circle(point, radius, color=color, alpha=1, edgecolor='black')
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
        calculator = PointSetRegistrationCalculator([dp.point for dp in self.draggable_points], self.target_points)
        initial_distance = calculator.calculate_average_distance()
        print(f"Initial average distance: {initial_distance}")

        final_distance = calculator.apply_icp()
        print(f"Final average distance after ICP: {final_distance}")

        for dp, new_point in zip(self.draggable_points, calculator.source_points):
            dp.point = new_point
            dp.circle.set_center(new_point)

        self.fig.canvas.draw_idle()

class PointSetRegistrationCalculator:
    def __init__(self, source_points, target_points):
        self.source_points = np.array(source_points)
        self.target_points = np.array(target_points)

    def calculate_average_distance(self):
        distances = np.sqrt(np.sum((self.source_points - self.target_points) ** 2, axis=1))
        return np.mean(distances)

    def apply_icp(self, iterations=10):
        for _ in range(iterations):
            centroid_source = np.mean(self.source_points, axis=0)
            centroid_target = np.mean(self.target_points, axis=0)
            source_centered = self.source_points - centroid_source
            target_centered = self.target_points - centroid_target
            H = source_centered.T @ target_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            t = centroid_target - R @ centroid_source
            self.source_points = (R @ source_centered.T).T + t
        return self.calculate_average_distance()

    

# Example usage
# translation
source_points = [(2, 1), (3, 4), (1, 3), (4, 3), (2, 2)]
target_points = [(7, 6), (8, 9), (6, 8), (9, 8), (7, 7)]



# rotaion
# source_points = [(3.00, 2.00), (2.50, 1.00), (2.00, 2.50), (1.00, 2.00), (1.50, 1.00)]
# target_points = [(6.5, 7.0), (7.0, 8.0), (7.5, 6.5), (8.5, 7.0), (8.0, 8.0)]


psr = PointSetRegistration(source_points, target_points)


psr_calculator = PointSetRegistrationCalculator(source_points, target_points)
initial_distance = psr_calculator.calculate_average_distance()
print(f"Initial average distance: {initial_distance}")

final_distance = psr_calculator.apply_icp()
print(f"Final average distance after ICP: {final_distance}")