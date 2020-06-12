import sys
from shutil import copyfile

print(sys.argv[1])
source_file_dev = 'data_list/dev_index_{}.json'.format(sys.argv[1])
destination_file_dev = 'dev.json'
source_file_train = 'data_list/trian_index_{}.json'.format(sys.argv[1])
destination_file_train = 'train.json'
copyfile(source_file_dev, destination_file_dev)
copyfile(source_file_train, destination_file_train)
