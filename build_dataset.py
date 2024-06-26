import pandas as pd
import random
def get_dataset(args):
    dataset = args.dataset

    # df = pd.read_csv("home/induct-gcn/data/"+dataset+".csv")
    # sentences = df['text'].tolist()
    # labels = df['label'].tolist()
    # train_or_test = df['train'].tolist()

    with open(f"home/induct-gcn/data/IOT2022.texts.txt", "r", encoding="latin1") as f:
        sentences = f.read().strip().split("\n")

    with open(f"home/induct-gcn/data/IOT2022.labels.txt", "r", encoding="latin1") as f:
        text2 = f.read().strip().split("\n")
        labels = []
        train_or_test=[]
        for i in range(len(text2)):
            txt1 = text2[i].split()
            labels.append(txt1[2])
            train_or_test.append(txt1[1])

    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(sentences)
    random.seed(randnum)
    random.shuffle(labels)
    random.seed(randnum)
    random.shuffle(train_or_test)


    # import pudb;pu.db
    train_sentences, train_labels, test_sentences, test_labels = [],[],[],[]
    original_train_size, original_test_size = 0, 0
    for i in range(len(train_or_test)):
        if train_or_test[i] == 'train':
            train_sentences.append(sentences[i])
            train_labels.append(labels[i])
            original_train_size += 1
        elif train_or_test[i] == 'test':
            test_sentences.append(sentences[i])
            test_labels.append(labels[i])
            original_test_size += 1
    if not args.easy_copy:
        print("There are",len(text2),"samples in",dataset)
        print("Original Training set size:",original_train_size)
        print("Original Test set size:",original_test_size)


    if args.train_size > 1:
        train_size = int(args.train_size)
    else:
        train_size = int(original_train_size*args.train_size)

    train_sentences = train_sentences[:train_size]
    train_labels = train_labels[:train_size]


    if args.test_size > 1:
        test_size = int(args.test_size)
    else:
        test_size = int(original_test_size*args.test_size)
    if test_size < original_test_size:
        test_sentences = test_sentences[:test_size]
        test_labels = test_labels[:test_size]
    else:
        if test_size+train_size > len(text2):
            raise SyntaxError('Not enough data, try use smaller train_size and test_size')
        test_sentences += train_sentences[:-(test_size-original_test_size)] 
        test_labels += train_labels[:-(test_size-original_test_size)] 

    if not args.easy_copy:
        print("Real Train Size:",train_size)
        print("Real Test Size:",test_size)
    return train_sentences+test_sentences, train_labels+test_labels, train_size, test_size