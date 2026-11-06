"""ML fault protection on the IEEE 123-node feeder: unseen grid layouts and false-data attacks.

Layers (each only imports from the ones above it):

    config        paths, sensors, constants
    grid          layouts, OpenDSS model, feeder graph, zones, path signatures
    simulation    random operating points + fault / normal events
    features      measurement noise + the 208-feature vector
    dataset       parallel generation, loading
    attacks       replay false-data injection
    models        baselines, topology-aware locator, defences, end-to-end scheme
    experiments   the three studies that produce results/
    figures, cli
"""
__version__ = "1.1.0"
