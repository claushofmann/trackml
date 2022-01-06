import pandas as pd
import numpy as np
#import load_data as ld
import matplotlib.pyplot as plt
from matplotlib import cm
import volumes as vs

# Creates visualizations of hits and tracks in the detector as found in my Thesis


def plot_layer_rings(no_layers = 10):
    fig, ax = plt.subplots()
    rad_array = [35, 74, 118, 176.5, 267, 366, 506, 700.5, 826, 1026]
    for radius_id in range(no_layers):
        radius = rad_array[radius_id]
        radius -= 1
        circle = plt.Circle((0, 0), radius, color='grey', fill=False)
        ax.add_artist(circle)


def prepare_layer_plot():
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1200, 1200)
    plt.ylim(-1200, 1200)


def filter_particle_hits(hit_group: pd.DataFrame):
    # We want to filter these particles where we have hits on every layer
    for layer in range(10):
        if not hit_group['layer_id'].isin([layer]).any():
            return False
    return True


if __name__ == '__main__':

    ld.DirectoryBuilder().kaggle_data().discrete_framified().attach()

    no_vis_particles = 15

    hits = ld.easy_load_event_hits(1000)
    truth = ld.easy_load_event_truth(1000)
    particles = ld.easy_load_event_particles(1000)

    merged = pd.merge(hits, truth[['particle_id', 'hit_id']], on='hit_id')
    # Remove noise
    merged = merged[merged['particle_id'] != 0]

    possible_hits = merged.groupby('particle_id').filter(filter_particle_hits)
    possible_part_ids = possible_hits['particle_id'].unique()
    possible_part_ids = pd.DataFrame(possible_part_ids, columns=['particle_id'])

    # Create visualisation with a sample of no_vis_particles
    vis_particles = possible_part_ids.sample(no_vis_particles)

    # Assign a distinct color to each particle
    colors = cm.rainbow(np.linspace(0, 1, no_vis_particles))
    np.random.shuffle(colors)
    vis_particles['color'] = colors.tolist()

    vis_hits = pd.merge(merged, vis_particles[['particle_id', 'color']], on='particle_id')
    vis_hits = vis_hits[  # Filter hits from only circular (barrel) layers
        (vis_hits['volume_id'] == vs.Pixel.BARREL.value['id']) | (vis_hits['volume_id'] == vs.LongStrip.BARREL.value['id'])
        | (vis_hits['volume_id'] == vs.ShortStrip.BARREL.value['id'])
    ]

    # Plot particles with color
    plot_layer_rings()
    prepare_layer_plot()
    plt.scatter(vis_hits['x'], vis_hits['y'], color=vis_hits['color'])
    plt.show()

    # Plot particles without color
    plot_layer_rings()
    prepare_layer_plot()
    plt.scatter(vis_hits['x'], vis_hits['y'])
    plt.show()
