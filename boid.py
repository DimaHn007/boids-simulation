'''
Simulation of the flocking behavior of birds (Boids model).

The three core rules of the Boids simulation are as follows:
    1. Separation - Keep a minimum distance between the boids.
    2. Alignment - Point each boid in the average direction of movement of
       its local flockmates.
    3. Cohesion - Move each boid toward the center of mass of its local
       flockmates.
'''

import sys, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm


width, height = 1000, 1000 # screen size


class Boids:
    """class that represents Boids simulation"""
    def __init__(self, N=50):
        """initialize the Boid simulation"""
        # --Initial position and velocities--
        # place all boids in approximately the center of the screen
        self.pos = [width/2, height/2] + 10*np.random.rand(2*N).reshape(N, 2)
        # normalized random velocities
        directions = 2*math.pi*np.random.rand(N)
        self.vel = np.array(list(zip(np.sin(directions), np.cos(directions))))
        self.N = N
        # minimum distance of approach
        self.minDist = 25.0
        # maximum magnitude of velocities calculated by "rules"
        self.maxRuleVel = 0.03
        # maximum magnitude of the final velocity
        self.maxVel = 2.0
        self.bird_count = N


    def tick(self, frameNum, pts, beak):
        """Update the simulation by one time step."""
        # get pairwise distances
        self.distMatrix = squareform(pdist(self.pos))
        # apply rules:
        self.vel += self.applyRules()
        self.limit(self.vel, self.maxVel)
        self.pos += self.vel
        self.applyBC()
        # update data
        pts.set_data(self.pos.reshape(2*self.N)[::2],
                     self.pos.reshape(2*self.N)[1::2])
        vec = self.pos + 10*self.vel/self.maxVel
        beak.set_data(vec.reshape(2*self.N)[::2],
                      vec.reshape(2*self.N)[1::2])


    def applyRules(self):
        # apply rule #1: Separation
        D = self.distMatrix < self.minDist
        vel = self.pos*D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, self.maxRuleVel)
        # distance threshold for alignment
        D = self.distMatrix < 50.0
        # apply rule #2: Alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.maxRuleVel)
        vel += vel2;
        # apply rule #3: Cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.maxRuleVel)
        vel += vel3
        return vel


    def limitVec(self, vec, maxVal):
        """limit the magnitide of the 2D vector"""
        mag = norm(vec)
        if mag > maxVal:
            vec[0], vec[1] = vec[0]*maxVal/mag, vec[1]*maxVal/mag
            

    def limit(self, X, maxVal):
        """limit the magnitude of 2D vectors in array X to maxValue"""
        for vec in X:
            self.limitVec(vec, maxVal)


    def applyBC(self, boundary_loss=0.8, boundary_speed=1, attraction_center=0.0001):
        deltaR = 2.0
        center = np.array([width / 2, height / 2])
        for i in range(self.N):
            coord = self.pos[i]
            vel = self.vel[i]
            if coord[0] > width + deltaR:
                vel[0] = -boundary_speed  # Встановлення швидкості в напрямку границі
                coord[0] = width + deltaR
            if coord[0] < -deltaR:
                vel[0] = boundary_speed
                coord[0] = -deltaR
            if coord[1] > height + deltaR:
                vel[1] = -boundary_speed
                coord[1] = height + deltaR
            if coord[1] < -deltaR:
                vel[1] = boundary_speed
                coord[1] = -deltaR
            # Додавання притягування до центру
            attraction = (center - coord) * attraction_center
            vel += attraction
        """
        Apply boundary conditions.

        deltaR  - provides a slight buffer, which allows the boid to move
                  slightly outside the tile before it starts coming back in
                  from the opposite direction, thus producing a better
                  visual effect
        
        deltaR = 2.0 
        for coord in self.pos:
            if coord[0] > width + deltaR:
                coord[0] = - deltaR
            if coord[0] < - deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = - deltaR
            if coord[1] < - deltaR:
                coord[1] = height + deltaR
        """

    def buttonPress(self, event):
        """event handler for matplotlib button presses"""
        # left-click to add a boid
        if event.button == 1:
            self.pos = np.concatenate((self.pos,
                                       np.array([[event.xdata, event.ydata]])),
                                       axis=0)
            # generate a random velocity
            angles = 2*math.pi*np.random.rand(1)
            v = np.array(list(zip(np.sin(angles), np.cos(angles))))
            self.vel = np.concatenate((self.vel, v), axis=0)
            self.N += 1
        # right-click to scatter (disturb) boids 
        elif event.button == 3:
            # add scattering velocity
            #self.vel += 0.1*(self.pos - np.array([[event.xdata, event.ydata]]))
            radius = 50  # Радіус, в якому повертаються пташки
            clicked_point = np.array([[event.xdata, event.ydata]])
            distances = np.linalg.norm(self.pos - clicked_point, axis=1)
            indices_within_radius = np.where(distances <= radius)[0]
            self.vel[indices_within_radius] += 0.1 * (self.pos[indices_within_radius] - clicked_point)
        elif event.button == 2:  # Середня кнопка миші - видалити boid
            self.removeBoid(event.xdata, event.ydata)

    def removeBoid(self, x, y):
        indices_to_remove = []
        for i in range(self.N):
            dist = np.sqrt((self.pos[i][0] - x) ** 2 + (self.pos[i][1] - y) ** 2)
            if dist < 10:  # Визначте власний радіус для видалення бід
                indices_to_remove.append(i)
        if self.N <= 1:
            print("This is the last bird. Unable to delete!!!")
        elif indices_to_remove:
            self.pos = np.delete(self.pos, indices_to_remove, axis=0)
            self.vel = np.delete(self.vel, indices_to_remove, axis=0)
            self.N -= len(indices_to_remove)


def tick(self, frameNum, pts, beak, boids):
    """update function for animation"""
    boids.tick(frameNum, pts, beak)
    self.bird_count = len(self.pos)
    return pts, beak


if __name__ == "__main__":
    boids = Boids()
    fig = plt.figure("Boids simulation")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.2)
    ax = plt.axes(xlim=(0, width), ylim=(0, height))
    ax.axis('off')
    rectangle = patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rectangle)
    pts, = ax.plot([], [], markersize=10, c='k', marker='o', ls='None')
    beak, = ax.plot([], [], markersize=4, c='r', marker='o', ls='None')
    
    text = ax.text(10, 10, f'Count birds: {boids.bird_count}', fontsize=12, color='green')

    def update_text(frameNum, pts, beak, boids, text):
        boids.tick(frameNum, pts, beak)
        bird_count = len(boids.pos)
        # Оновлення тексту вже існуючого об'єкту text
        text.set_text(f'Count birds: {bird_count}')
        return pts, beak, text

    anim = animation.FuncAnimation(fig, update_text, fargs=(pts, beak, boids, text), interval=50)
    # add a "button press" event handler
    cid = fig.canvas.mpl_connect('button_press_event', boids.buttonPress)
    plt.show()