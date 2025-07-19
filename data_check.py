import pickle
from pprint import pprint

cache_path = r'C:\Users\andus\Desktop\my_work\BOAZ 23기\ADV\boaz_sample2\projects\checkpoints\data_cache.pkl'

with open(cache_path, 'rb') as f:
    train_meta, train_labels, test_meta, test_labels = pickle.load(f)

print(train_meta)
pprint(train_meta)
print(len(train_meta))
print()

print(train_labels)
pprint(train_labels)
print(len(train_labels))
print()

print(test_meta)
pprint(test_meta)
print(len(test_meta))
print()

print(test_labels)
pprint(test_labels)
print(len(test_labels))
print()

