import os
import os.path as osp
from PIL import Image
from torchvision.transforms import Resize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

rsz = Resize((512,512))
rootDir = 'labeled-images/'
outDir = 'data/kvasir/'
os.makedirs(outDir, exist_ok=True)

for dirName, subdirList, fileList in os.walk(rootDir):
    print(dirName)
    for fname in fileList:
        if 'jpg' in fname:
            img = Image.open(osp.join(dirName, fname))
            img_res = rsz(img)
            img_res.save(osp.join(outDir,fname))

df_all = pd.read_csv('labeled-images/image-labels.csv')
findings = df_all['Finding'].values

findings_list = list(np.unique(findings))
findings_to_class = dict(zip(findings_list, np.arange(len(findings_list))))
class_to_findings = dict(zip(np.arange(len(findings_list)),findings_list))

images = df_all['Video file'].values
df_all['category'] = df_all.Finding.replace(findings_to_class)
df_all.drop(['Organ', 'Classification'], axis=1, inplace=True)
df_all.columns = ['image_id', 'finding_name', 'label']

im_list = df_all.image_id.values
im_list = [n+'.jpg' for n in im_list]
df_all.image_id = im_list

num_ims = len(df_all)
df_train, df_test = train_test_split(df_all, test_size=num_ims//5, random_state=0, stratify=df_all.label)
df_train, df_val = train_test_split(df_train, test_size=num_ims//5, random_state=0, stratify=df_train.label)

df_train.to_csv('data/train_kvasir.csv', index=None)
df_val.to_csv('data/val_kvasir.csv', index=None)
df_test.to_csv('data/test_kvasir.csv', index=None)
