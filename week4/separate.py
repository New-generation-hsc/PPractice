ROOT_PATH = "/home/aistudio/data/data274"

def split_dataset():
    # if the dataset is splited already, then nothing to do
    os.chdir(ROOT_PATH)
    if os.path.exists('storetag'):
        return
    os.mkdir('storetag')
    os.chdir('storetag')
    os.mkdir('train')
    os.mkdir('test')
    images = {}
    for line in open(ROOT_PATH + "/train/train.txt", 'r'):
        row_list = line.split(',')
        label = row_list[1]
        images.setdefault(label, set)
        images[label].add(row_list[0])
    train_images = {}
    test_images = {}
    for label in images:
        img_list = list(images[label])
        random.shuffle(img_list)
        pivot = len(img_list) * 4 / 5
        train_images[label] = img_list[:pivot]
        test_images[label] = img_list[pivot:]
    
    print("start copy....")
    os.chdir('train')
    idx = 0
    for label in train_images:
        if not os.path.exists(label):
            os.mkdir(label)
        for img in train_images[label]:
            idx += 1
            print("copy {}th image....".format(idx))
            shutil.copy(ROOT_PATH + "/train/train/" + img, label + "/" + img)
            
    os.chdir('../test')
    for label in test_images:
        if not os.path.exists(label):
            os.mkdir(label)
        for img in test_images[label]:
            idx += 1
            print("copy {}th image....".format(idx))
            shutil.copy(ROOT_PATH + "/train/train/" + img, label + "/" + img)