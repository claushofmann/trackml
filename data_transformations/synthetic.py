import numpy as np
from volumes import *
import pandas as pd


def synthesize_data(no_particles):

    thetas = np.random.uniform(0, np.pi, no_particles)
    phis = np.random.uniform(-np.pi, np.pi, no_particles)

    dfs = []

    for volume in all_volumes:
        geometry = volume.value['layer_geometry']
        orientation = geometry['orientation']
        for i, layer in enumerate(geometry['layers']):
            layer_df = pd.DataFrame(columns=['x', 'y', 'z', 'r', 'particle_id', 'hit_id', 'volume_id', 'layer_id', 'weight'])
            layer_df['particle_id'] = list(range(no_particles))
            layer_df['volume_id'] = [volume.value['id'] for _ in range(no_particles)]
            layer_df['layer_id'] = [2 * (i + 1) for _ in range(no_particles)]
            layer_df['weight'] = 1 / no_particles
            layer_df[geometry['layer_dim']] = layer
            if orientation == 'z':
                layer_df['z'] = np.tan(thetas) * layer_df['r']
            if orientation == 'r':
                layer_df['r'] = layer_df['z'] / np.tan(thetas)
            layer_df['x'] = np.cos(phis) * layer_df['r']
            layer_df['y'] = np.sin(phis) * layer_df['r']
            try:
                selected_layer_df = layer_df[layer_df[orientation].between(*volume.value['geometry'][orientation])]
            except:
                print('Hello')
            selected_layer_df = selected_layer_df.sample(frac=1)
            dfs.append(selected_layer_df)

    df = pd.concat(dfs)
    df['hit_id'] = np.random.choice(10000000, replace=False, size=len(df.index))

    df[['tx', 'ty', 'tz']] = df[['x', 'y', 'z']]

    hits = df[['x', 'y', 'z', 'hit_id', 'volume_id', 'layer_id']]
    truth = df[['tx', 'ty', 'tz', 'hit_id', 'particle_id', 'weight']]

    empty = ['this', 'df', 'is', 'empty']

    return hits, pd.DataFrame([empty], columns=empty), pd.DataFrame([empty], columns=empty), truth

def make_synthetic_file(name, no_particles):

    hits, particles, cells, truth = synthesize_data(no_particles)

    hits.to_csv(name + '-hits.csv', index=False)
    truth.to_csv(name + '-truth.csv', index=False)
    particles.to_csv(name + '-particles.csv', index=False)
    cells.to_csv(name + '-cells.csv', index=False)

if __name__ == '__main__':
    a,b,c,d = synthesize_data(10000)
    print('Done')
