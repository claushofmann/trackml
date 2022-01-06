import numpy as np



def make_theta_file(name, hits, cells, particles, truth):
    def theta_computation(hits, x_name, y_name, z_name, theta_name):
        r = np.sqrt(hits[x_name]**2 + hits[y_name]**2)
        theta = np.arctan2(r, hits[z_name])
        theta = theta / (2 * np.pi)
        hits[theta_name] = theta

    transf_hits = hits
    theta_computation(transf_hits, 'x', 'y', 'phi')
    transf_hits.to_csv(name + '-hits.csv', index=False)

    transf_particles = particles
    theta_computation(transf_particles, 'vx', 'vy', 'vphi')
    transf_particles.to_csv(name + '-particles.csv', index=False)

    transf_truth = truth
    theta_computation(transf_truth, 'tx', 'ty', 'tphi')
    transf_truth.to_csv(name + '-truth.csv', index=False)

    cells.to_csv(name + '-cells.csv', index=False)