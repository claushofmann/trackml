from enum import Enum

# Contains information about the detector structure


class Pixel(Enum):
    BARREL = {
        'id': 8,
        'no_layers': 4,
        'geometry': {
            'x': [-176, 176],
            'y': [-176, 176],
            'z': [-492, 492]
        },
        'layer_geometry': {
            'orientation': 'z',
            'layer_dim': 'r',
            'layers': [32, 72, 116, 172]
        },
        'reversed': False}
    NEGATIVE_EC = {'id': 7,
                   'no_layers': 7,
                   'geometry': {
                       'x': [-176, 176],
                       'y': [-176, 176],
                       'z': [-1503, -596],
                       'r': [32, 172]
                   },
                   'layer_geometry': {
                       'orientation': 'r',
                       'layer_dim': 'z',
                       'layers': [-1500, -1300, -1100, -960, -820, -700, -600]
                   },
                   'reversed': True}
    POSITIVE_EC = {'id': 9,
                   'no_layers': 7,
                   'geometry': {
                       'x': [-176, 176],
                       'y': [-176, 176],
                       'z': [596, 1503],
                       'r': [32, 172]
                   },
                   'layer_geometry': {
                       'orientation': 'r',
                       'layer_dim': 'z',
                       'layers': [600, 700, 820, 960, 1100, 1300, 1500]
                   },
                   'reversed': False}


class ShortStrip(Enum):
    BARREL = {'id': 13,
              'no_layers': 4,
              'geometry': {
                  'x': [-667, 667],
                  'y': [-667, 667],
                  'z': [-1085, 1085]
              },
              'layer_geometry': {
                'orientation': 'z',
                'layer_dim': 'r',
                'layers': [260, 360, 500, 660]
              },
              'reversed': False}
    NEGATIVE_EC = {'id': 12,
                   'no_layers': 6,
                   'geometry': {
                       'x': [-702, 700],
                       'y': [-702, 700],
                       'z': [-2956, -1213],
                       'r': [260, 660]
                   },
                   'layer_geometry': {
                       'orientation': 'r',
                       'layer_dim': 'z',
                       'layers': [-2950, -2550, -2150, -1800, -1500, -1220]
                   },
                   'reversed': True}
    POSITIVE_EC = {'id': 14,
                   'no_layers': 6,
                   'geometry': {
                       'x': [-702, 700],
                       'y': [-702, 700],
                       'z': [1213, 2956],
                       'r': [260, 660]
                   },
                   'layer_geometry': {
                       'orientation': 'r',
                       'layer_dim': 'z',
                       'layers': [1220, 1500, 1800, 2150, 2550, 2950]
                   },
                   'reversed': False}


class LongStrip(Enum):
    BARREL = {'id': 17,
              'no_layers': 2,
              'geometry': {
                  'x': [-1027, 1027],
                  'y': [-1027, 1027],
                  'z': [-1080, 1080]
              },
              'layer_geometry': {
                  'orientation': 'z',
                  'layer_dim': 'r',
                  'layers': [820, 1020]
              },
              'reversed': False}
    NEGATIVE_EC = {'id': 16,
                   'no_layers': 6,
                   'geometry': {
                       'x': [-1016, 1016],
                       'y': [-1016, 1016],
                       'z': [-2957, -1212],
                       'r': [820, 1020]
                   },
                   'layer_geometry': {
                       'orientation': 'r',
                       'layer_dim': 'z',
                       'layers': [-2950, -2550, -2150, -1800, -1500, -1220]
                   },
                   'reversed': True}
    POSITIVE_EC = {'id': 18,
                   'no_layers': 6,
                   'geometry': {
                       'x': [-1016, 1016],
                       'y': [-1016, 1016],
                       'z': [1212, 2957],
                       'r': [820, 1020]
                   },
                   'layer_geometry': {
                       'orientation': 'r',
                       'layer_dim': 'z',
                       'layers': [1220, 1500, 1800, 2150, 2550, 2950]
                   },
                   'reversed': False}


all_volumes = (Pixel.BARREL, Pixel.NEGATIVE_EC, Pixel.POSITIVE_EC, ShortStrip.BARREL, ShortStrip.NEGATIVE_EC, ShortStrip.POSITIVE_EC, LongStrip.BARREL, LongStrip.NEGATIVE_EC, LongStrip.POSITIVE_EC)


vol_id_index = {}


def get_layer_ids(volume):
    if volume.value['reversed']:
        return [(idx + 1) * 2 for idx in reversed(range(volume.value['no_layers']))]
    else:
        return [(idx + 1) * 2 for idx in range(volume.value['no_layers'])]


def __insert(detector):
    for vol in detector:
        vol_id_index[vol.value['id']] = vol


__insert(Pixel)
__insert(ShortStrip)
__insert(LongStrip)