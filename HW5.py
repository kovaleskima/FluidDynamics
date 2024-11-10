import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.animation as animation
from typing import Tuple, List

class Vortex: 
    def __init__(self, gamma: float, center_x: float, center_y: float):
        self.gamma = gamma
        self.center_x = center_x
        self.center_y = center_y

    def get_polar(self, center_x: float, center_y: float, coord_x: int, coord_y: int) -> Tuple[float, float]:
        r = np.sqrt((coord_x - center_x)**2 + (coord_y - center_y)**2)
        theta = np.arctan2((coord_y - center_y), (coord_x - center_x))
        return (r, theta)

    def apply_phantom_vortex(self, coord_x: int, coord_y: int, copy_x: int, copy_y: int) -> Tuple[float, float]:
        phantom_gamma = -self.gamma
        phantom_r, phantom_theta = self.get_polar(copy_x, copy_y, coord_x, coord_y)
        u_theta_x = 0
        u_theta_y = 0
        if 0 < phantom_r < 10:
            u_theta = phantom_gamma/(2*np.pi*phantom_r)
            u_theta_x = -u_theta*np.sin(phantom_theta)
            u_theta_y = u_theta*np.cos(phantom_theta)
        
        return (u_theta_x, u_theta_y)
    

    def generate_vectorfield(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # GIVEN a vectorfield, generate the change in the vectorfield caused by this vortex
        U_matrix = np.zeros_like(x)
        V_matrix = np.zeros_like(y)
        for i in range(len(x)):
            for j in range(len(x[i])):
                r, theta = self.get_polar(self.center_x, self.center_y, i, j)
                u_theta_x = 0.0
                u_theta_y = 0.0
                if 0 < r and r < 10:
                    u_theta = self.gamma/(2*np.pi*r)
                    u_theta_x = -u_theta*np.sin(theta)
                    u_theta_y = u_theta*np.cos(theta)
                

                if i < 10:
                    u_theta_x_p, u_theta_y_p = self.apply_phantom_vortex(i, j, -self.center_x, self.center_y)
                    u_theta_x += u_theta_x_p
                    u_theta_y += u_theta_y_p
                if i > len(x) - 11:
                    u_theta_x_p, u_theta_y_p = self.apply_phantom_vortex(i, j, 2*len(x) - self.center_x, self.center_y)
                    u_theta_x += u_theta_x_p
                    u_theta_y += u_theta_y_p
                if j < 10:
                    u_theta_x_p, u_theta_y_p = self.apply_phantom_vortex(i, j, self.center_x, -self.center_y)
                    u_theta_x += u_theta_x_p
                    u_theta_y += u_theta_y_p
                if j > len(y[i]) - 11:
                    u_theta_x_p, u_theta_y_p = self.apply_phantom_vortex(i, j, self.center_x,  2*len(x[i])-self.center_y)
                    u_theta_x += u_theta_x_p
                    u_theta_y += u_theta_y_p


                U_matrix[i][j] = u_theta_x
                V_matrix[i][j] = u_theta_y
        return (U_matrix, V_matrix)

class Field:
    def __init__(self, x_dim: int, y_dim: int, vortices: List[Vortex]):
        self.vortices = vortices
        self.x = np.zeros((x_dim, y_dim), dtype=float)
        self.y = np.zeros((x_dim, y_dim), dtype=float)

    def move_centers(self, dt: float):
        for i in range(len(self.vortices)):
            u_theta_x = 0.0
            u_theta_y = 0.0
            for j in range(len(self.vortices)):
                if i != j:
                    #Move vortices[i] wrt. vortices[j]
                    r, theta = self.vortices[j].get_polar(self.vortices[j].center_x, self.vortices[j].center_y,
                                                           self.vortices[i].center_x, self.vortices[i].center_y)
                    u_theta = self.vortices[j].gamma/(2*np.pi*r)
                    u_theta_x += -u_theta*np.sin(theta)
                    u_theta_y += u_theta*np.cos(theta)
                
            #Move vortices[i] wrt. phantom vortices
            if self.vortices[i].center_x < 10:
                u_theta_x_p, u_theta_y_p = self.vortices[i].apply_phantom_vortex(self.vortices[i].center_x, self.vortices[i].center_y, -self.vortices[i].center_x, self.vortices[i].center_y)
                u_theta_x += u_theta_x_p
                u_theta_y += u_theta_y_p
            if self.vortices[i].center_x > len(self.x) - 11:
                u_theta_x_p, u_theta_y_p = self.vortices[i].apply_phantom_vortex(self.vortices[i].center_x, self.vortices[i].center_y, 2*len(self.x) - self.vortices[i].center_x, self.vortices[i].center_y)
                u_theta_x += u_theta_x_p
                u_theta_y += u_theta_y_p
            if self.vortices[i].center_y < 10:
                u_theta_x_p, u_theta_y_p = self.vortices[i].apply_phantom_vortex(self.vortices[i].center_x, self.vortices[i].center_y, self.vortices[i].center_x, -self.vortices[i].center_y)
                u_theta_x += u_theta_x_p
                u_theta_y += u_theta_y_p
            if self.vortices[i].center_y > len(self.x[0]) - 11:
                u_theta_x_p, u_theta_y_p = self.vortices[i].apply_phantom_vortex(self.vortices[i].center_x, self.vortices[i].center_y, self.vortices[i].center_x, 2*len(self.x[0])-self.vortices[i].center_y )
                u_theta_x += u_theta_x_p
                u_theta_y += u_theta_y_p
                    
            u_theta_x_p = 0.0
            u_theta_y_p = 0.0
            self.vortices[i].center_x += u_theta_x * dt
            self.vortices[i].center_y += u_theta_y * dt

    def move_timestep(self, dt: float):
        # 1. Calculate vectorfields of all vortices
        U_matrix = np.zeros_like(self.x, dtype=float)
        V_matrix = np.zeros_like(self.y, dtype=float)
        for vort in self.vortices:
            U, V = vort.generate_vectorfield(self.x, self.y)
            # 2. Sum them up on the overall vectorfield
            U_matrix += U
            V_matrix += V

        self.x += U_matrix
        self.y += V_matrix

        # 3. Move the vortices
        self.move_centers(dt)

# Set up the grid for the vector field
vortex1 = Vortex(1, 15, 20)
vortex2 = Vortex(-1, 30, 20)
vortex3 = Vortex(-1, 20, 30)
vortex4 = Vortex(1, 30, 30)
field = Field(50, 50, [vortex1, vortex2, vortex3, vortex4])
U = field.x
V = field.y
norm = np.sqrt(U**2 + V**2)

# Set up the figure and axis
fig, ax = plt.subplots()
scatter = ax.scatter([vortex1.center_x, vortex2.center_x, vortex3.center_x, vortex4.center_x], 
                     [vortex1.center_y, vortex2.center_y, vortex3.center_y, vortex4.center_y], color=['r', 'b', 'g', 'm'])
ax.set_xlim(-1, 51)
ax.set_ylim(-1, 51)
ax.set_title("Animation")
mesh_x, mesh_y = np.mgrid[:len(field.x), :len(field.x[0])]
quiver = ax.quiver(mesh_x, mesh_y, U, V, norm, cmap='nipy_spectral', scale=50, width=0.002 )

# Add a colorbar to show the scale of magnitudes
cbar = fig.colorbar(quiver, ax=ax)
cbar.set_label('Magnitude')

# Animation function to update U and V at each frame
def animate(frame):
    # update field
    field.move_timestep(1.0)
    U = field.x
    V = field.y
    norm = np.sqrt(U**2 + V**2)

    quiver.set_UVC(U/norm,V/norm, norm)
    scatter.set_offsets([(vortex1.center_x, vortex1.center_y), (vortex2.center_x, vortex2.center_y), 
                         (vortex3.center_x, vortex3.center_y), (vortex4.center_x, vortex4.center_y)])
    
    return quiver, scatter

ani = animation.FuncAnimation(fig, animate, frames=range(100), interval=1, blit=False, cache_frame_data=False, repeat=True)
plt.show()