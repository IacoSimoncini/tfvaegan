#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

def get_where(label, loc, val=True):
    arr = np.isin(label, loc)
    result = np.where(arr == val)
    return result[0]

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def split_train_test(self, test_train, label):
        count = np.zeros(len(self.seenclasses) + len(self.unseenclasses))
        train = []
        test = []
        for i in test_train:
            if  count[label[i]] != 4:
                train.append(i)
                count[label[i]] = count[label[i]] + 1
            else:
                test.append(i)
                count[label[i]] = 0
        return np.array(train), np.array(test)


    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
          

        attribute = matcontent['att']
        #UNCOMMENT THE CODE FROM HERE IF YOU USE MERGING ATTRIBUTES
       """ index1 = [137, 142, 271]    #CHANGE THIS ARRAY WITH THE FIRST 3 ATTRIBUTES YOU WANT TO MERGE
        index2 =  [108, 200, 42]    #CHANGE THIS ARRAY WITH THE SECOND 3 ATTRIBUTES YOU WANT TO MERGE
        col1 = []
        col2 = []
        for i in index1:
            col1.append(attribute[i])
        for i in index2:
            col2.append(attribute[i])
        total=index1+index2
        attribute = np.delete(attribute, total, axis=0)   
        col1 = np.array(col1)
        col2 = np.array(col2)
        mean1 = np.mean(col1, axis=0)
        mean2 = np.mean(col2, axis=0)
        attribute = np.transpose(attribute)
        attribute = np.c_[attribute, mean1]
        attribute = np.c_[attribute, mean2]
        attribute = attribute.T
"""
        #STOP UNCOMMENTING 
        
        #UNCOMMENT HERE FOR DELETING ATTRIBUTES
        """
        attribute = np.delete(attribute, [137, 142, 271, 108, 200, 42], axis=0) #CHANGE THIS ARRAY WITH THE ATTRIBUTES YOU WANT TO DELETE
        
        """
        self.attribute = torch.from_numpy(attribute.T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))


        
        all_classes = np.arange(start=0, stop=opt.nclass_all - 1, step=1)

        #CHANGE HERE WITH THE ARRAY OF THE UNSEEN CLASS OF THE SPLITS YOU WANT TO USE
        unseen_classes = np.array([6,  18,  20,  28,  33,  35,  49,  55,  61,  67,  68,  71,  78,  79,  86,  87,  90,  94,
                                    97,  99, 103, 107, 115, 119, 121, 123, 124, 128, 138, 140, 141, 149, 151, 156, 158, 159,
                                    165, 166, 170, 173, 175, 178, 181, 184, 186, 188, 190, 191, 192, 194])  

        test_unseen_loc = get_where(label, unseen_classes)
        np.random.shuffle(test_unseen_loc)
        self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                
        seen_classes = get_where(all_classes, unseen_classes, False)
        test_train = get_where(label, seen_classes)
        np.random.shuffle(test_train)

        self.seenclasses = torch.from_numpy(seen_classes)
        self.unseenclasses = torch.from_numpy(unseen_classes)

        train, test = self.split_train_test(test_train, label)
        self.train_label = torch.from_numpy(label[train]).long()

        self.train_feature = torch.from_numpy(feature[train]).float()
        self.test_seen_feature = torch.from_numpy(feature[test]).float()
        self.test_seen_label = torch.from_numpy(label[test]).long()

        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 


    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att


   
