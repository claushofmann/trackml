"""
This package contains data transformations which are used in the training ML-Pipeline
"""


def get_event_name(event_id):
    return 'event' + str(event_id).zfill(9)