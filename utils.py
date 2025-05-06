def get_sequence_start_indices(df, sequence_length=30, horizon=1, stride=1, group_col='time_group'):
    indices = []
    group_to_indices = {}

    for idx, group in zip(df.index, df[group_col]):
        group_to_indices.setdefault(group, []).append(idx)

    for group, idxs in group_to_indices.items():
        if len(idxs) < sequence_length + horizon:
            continue

        idxs = sorted(idxs)
        max_start = len(idxs) - sequence_length - horizon + 1
        for start in range(0, max_start, stride):
            indices.append(idxs[start])

    return indices
