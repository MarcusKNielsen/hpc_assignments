import numpy as np
import pyvista as pv

# Load the VTK file
#filename = "poisson_gs_100_30000_0.010000_.vtk"
filename = "poisson_j_100_30000_0.001000_.vtk"
path = "/home/max/Documents/DTU/HighPerformanceComputing/hpc_assignments/assignment2/plots/"
grid = pv.read(path + filename)

print("Point data arrays:", grid.point_data.keys())
print("Cell data arrays:", grid.cell_data.keys())

# Define scalar values for isosurfaces
isosurface_values = np.arange(1, 50, 1)

# Generate isosurfaces
contours = grid.contour(isosurfaces=isosurface_values, scalars="gray")

# Set up the PyVista plotter
pl = pv.Plotter(window_size=[600, 600])

# Add the isosurface mesh to the plotter with a custom color bar title
pl.add_mesh(
    contours,
    cmap="viridis",
    opacity=0.7,
    scalar_bar_args={
        "title": "Solution: u(x,y,z)",
        "title_font_size": 17,
        "label_font_size": 15,
        "color": "black"
    }
)

# Set the initial view
pl.view_zy()

# Set the up direction for the camera (ensuring horizontal rotation)
pl.set_viewup([0, 1, 0])  # The y-axis is up for horizontal rotation

# Show the final plot
pl.show()

# Show the plot
pl.show()







