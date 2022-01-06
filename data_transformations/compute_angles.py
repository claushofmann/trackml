import numpy as np



def make_angles_file(name, hits, cells, particles, truth):
    def phi_computation(hits, x_name, y_name, phi_name):
        arctan = np.arctan2(hits[y_name], hits[x_name])
        angle = arctan / (2 * np.pi)
        hits[phi_name] = angle

    def theta_computation(hits, x_name, y_name, z_name, theta_name):
        r = np.sqrt(hits[x_name]**2 + hits[y_name]**2)
        theta = np.arctan2(r, hits[z_name])
        theta = (theta / np.pi) - 0.5
        hits[theta_name] = theta

    transf_hits = hits
    phi_computation(transf_hits, 'x', 'y', 'phi')
    theta_computation(transf_hits, 'x', 'y', 'z', 'theta')
    transf_hits.to_csv(name + '-hits.csv', index=False)

    transf_particles = particles
    #phi_computation(transf_particles, 'vx', 'vy', 'vphi')
    #theta_computation(transf_particles, 'vx', 'vy', 'vz', 'vtheta')
    transf_particles.to_csv(name + '-particles.csv', index=False)

    transf_truth = truth
    phi_computation(transf_truth, 'tx', 'ty', 'tphi')
    theta_computation(transf_truth, 'tx', 'ty', 'tz', 'ttheta')
    transf_truth.to_csv(name + '-truth.csv', index=False)

    cells.to_csv(name + '-cells.csv', index=False)