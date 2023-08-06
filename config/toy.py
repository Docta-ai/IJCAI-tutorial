
details = False
num_classes = 2

hoc_cfg = dict(
    max_step = 1501, 
    T0 = None, 
    p0 = None, 
    lr = 0.1, 
    num_rounds = 50, 
    sample_size = 10000,
    already_2nn = True,
    device = 'cpu'
)


detect_cfg = dict(
    num_epoch = 51,
    sample_size = 10000,
    k = 10,
    name = 'simifeat',
    method = 'rank'
)