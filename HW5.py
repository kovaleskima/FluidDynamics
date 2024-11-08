import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##################
global center1
global center2
global u_theta1
global u_theta2
global dt

#############
# FUNCTIONS #
#############

def vortex_maker(x, y, Gamma):
    center = np.array([x,y])
    # REMEMBER TO DIVIDE BY R LATER
    u_theta = Gamma/(2*np.pi)
    return center, u_theta

# Generate a velocity vector (U,V) at any point wrt any individual vortex
# VECTORIZED
def get_velocity(x, y, center, u_theta):
    # Calculate distance from center
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan2(y-center[1], x-center[0])
    
    # Initialize u_theta divided by r, handle r == 0 directly
    with np.errstate(divide='ignore', invalid='ignore'):
        u_theta = np.where(r != 0, u_theta / r, 0)  # Avoid division by zero


    # Calculate velocity components
    U = -u_theta * np.sin(theta)
    V = u_theta * np.cos(theta)

    return U, V


##################
# INITIALIZATION #
##################

X, Y = np.mgrid[:50,:50]
center1, u_theta1 = vortex_maker(15, 15, 1)
center2, u_theta2 = vortex_maker(1, 1, 1)
dt = 10
steps = 250

# Set velocity for the first set of iterations
U1, V1 = get_velocity(X,Y,center1, u_theta1)
U2, V2 = get_velocity(X,Y, center2, u_theta2)

U = U1 + U2
V = V1 + V2

fig, ax = plt.subplots(1,1)
scatter = ax.scatter([center1[0], center2[0]], [center1[1],center2[1]], color=['r','b'])
Q = ax.quiver(X, Y, U, V, pivot='mid', color='grey', alpha=0.2, units='inches', scale=30, minshaft=1)

ax.set_xlim(0,50)
ax.set_ylim(0,50)

def update_quiver(num, Q, X, Y, U, V, center1, center2, u_theta1, u_theta2, dt):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    #define velocity here again
    U1, V1 = get_velocity(X,Y,center1, u_theta1)
    U2, V2 = get_velocity(X,Y, center2, u_theta2)

    U += U1 + U2
    V += V1 + V2

    center1 = center1 + [dt*U[center1[0],center1[1]], dt*V[center1[0], center1[1]]]
    center2 = center2 + [dt*U[center2[0],center2[1]], dt*V[center2[0], center2[1]]]

    Q.set_UVC(U,V)
    scatter.set_offsets([tuple(center1),tuple(center2)])

    return Q, scatter

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y, U, V, center1, center2, u_theta1, u_theta2, dt),
                               interval=50, blit=True, frames=steps, cache_frame_data=False, repeat=False)
fig.tight_layout()
plt.show()
