
import array

import zipfile
import npyfile
import emlearn_trees
import timebased

def argmax(arr):
    idx_max = 0
    value_max = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > value_max:
            value_max = arr[i]
            idx_max = i

    return idx_max

def har_load(path):
    print(f"loading {path}")
    n_features = timebased.N_FEATURES
    with npyfile.Reader(path) as data:
        shape = data.shape
        print(shape)
        assert len(shape) == 2, shape
        print(data.itemsize)
        assert data.itemsize == 2
        assert data.typecode == 'h'
        print(n_features)
        data_chunk = n_features
        sample_count = 0
        data_chunks = data.read_data_chunks(data_chunk)
        for arr in data_chunks:
            yield arr
            sample_count += 1



def main():

    model = emlearn_trees.new(10, 1000, 10)
    dataset = 'uci_har'
    data_path = '2000-01-01T000051_jumpingjack.npy'
    data_path = "har-10.npy"
    model_path = f'{dataset}.trees.csv'
    with open(model_path, 'r') as f:
        emlearn_trees.load_model(model, f)
    out = array.array('f', range(model.outputs()))
    print('har-run-load', data_path)
    for features in har_load(data_path):
        model.predict(features, out)
        result = argmax(out)
        print(result)

if __name__ == '__main__':
    main()
    # {"LAYING": 0, "SITTING": 1, "STANDING": 2, "WALKING": 3, "WALKING_DOWNSTAIRS": 4, "WALKING_UPSTAIRS": 5}
