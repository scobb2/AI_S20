# Generator suitable for training by presenting historical data sequences
# in order to predict a data point in the future, e.g. weather or stocks.
# 
# data -- A tensor of shape (time_span, features) providing feature data across
# some time span
#
# lookback -- Number of timesteps into past to look back for a training set
#
# forecast -- Number of timesteps into the future to predict target data
#
# min_idx, max_idx -- range within |data| to use
#
# trg_idx -- Index of target datum to predict within the list of data per timestep
def generator(data, lookback, forecast, min_idx, max_idx, trg_idx,
    batch_size=128):

    if max_index is None:
        max_index = len(data) - forecast - 1
    sample_point = min_index + lookback

    while 1:
        # Reset sample_point if there aren't enough timesteps ahead for a batch
        if sample_point + batch_size >= max_index:
            sample_point = min_index + lookback
        
        # Specific list of times in |data| to use as our batch
        times = np.arange(sample_point, min(sample_point + batch_size, max_index))
        sample_point += len(times)

        # Zero out one input/output batch
        samples = np.zeros((len(times), lookback, data.shape[-1]))
        targets = np.zeros((len(times),))

        for j, time in enumerate(times):
            indices = range(time - lookback, time)
            samples[j] = data[indices]
            targets[j] = data[time + forecast][trg_idx]

        yield samples, targets

    
    
    
    for s, t in generator(......):
