# device/micropython code

# Dans Thonny > View > Files
# Dans ESP32/lib/xtensawin_6.3 cliquer droit sur emlearn_trees.mpy > upload to

import emlearn_trees
import array

def argmax(arr):
    idx_max = 0
    value_max = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > value_max:
            value_max = arr[i]
            idx_max = i
    return idx_max

# max nb tree = 3
# max leaf = 2 (true ou false)
# max nodes (all tree) = 30
model = emlearn_trees.new(3, 30, 2)

with open('xor_model.csv', 'r') as f:
    emlearn_trees.load_model(model, f)

# run it
max_val = (2**15-1) # 1.0 as int16
examples = [
    array.array('h', [0, 0]),
    array.array('h', [max_val, max_val]),
    array.array('h', [0, max_val]),
    array.array('h', [max_val, 0]),
]

out = array.array('f', range(model.outputs()))
for ex in examples:
    model.predict(ex, out)
    result = argmax(out)
    print(ex, out, result)

