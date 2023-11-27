from dronebuddylib.offline.atoms.resources.matching.TuneAPI import tune


def update_memory():
    """update the memory by retraining the model (it may take several minutes)
    """
    tune()
    return
