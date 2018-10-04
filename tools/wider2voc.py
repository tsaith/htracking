
import os,h5py,cv2,sys,shutil
import numpy as np
from xml.dom.minidom import Document
rootdir="./"
convet2yoloformat=False
convert2vocformat=True
resized_dim=(48, 48)


#最小取40大小的脸，并且补齐
minsize2select=40
usepadding=True
 
datasetprefix="/home/andrew/projects/datasets/face_datasets/WIDER/widerface"#
def gen_hdf5():
    imgdir=rootdir+"/WIDER_train/images"
    gtfilepath=rootdir+"/wider_face_split/wider_face_train_bbx_gt.txt"
    index =0
    with open(gtfilepath,'r') as gtfile:
        faces=[]
        labels=[]
        while(True ):#and len(faces)<10
            imgpath=gtfile.readline()[:-1]
            if(imgpath==""):
                break;
            print(index, imgpath)
            img=cv2.imread(imgdir+"/"+imgpath)
            numbbox=int(gtfile.readline())
            bbox=[]
            for i in range(numbbox):
                line=gtfile.readline()
                line=line.split()
                line=line[0:4]               
                if(int(line[3])<=0 or int(line[2])<=0):
                    continue
                bbox=(int(line[0]),int(line[1]),int(line[2]),int(line[3]))
                face=img[int(line[1]):int(line[1])+int(line[3]),int(line[0]):int(line[0])+int(line[2])]
                face=cv2.resize(face, resized_dim)
                faces.append(face)
                labels.append(1)
                cv2.rectangle(img,(int(line[0]),int(line[1])),(int(line[0])+int(line[2]),int(line[1])+int(line[3])),(255,0,0))
            #cv2.imshow("img",img)
            #cv2.waitKey(1)
            index=index+1
        faces=np.asarray(faces)
        labels=np.asarray(labels)
        f=h5py.File('train.h5','w')
        f['data']=faces.astype(np.float32)
        f['label']=labels.astype(np.float32)
        f.close()
def viewginhdf5():
    f = h5py.File('train.h5','r') 
    f.keys()
    faces=f['data'][:]
    for face in faces:
        face=face.astype(np.uint8)
        cv2.imshow("img",face)
        cv2.waitKey(1)
    f.close()
 
def convertimgset(img_set="train"):
    imgdir=rootdir+"/WIDER_"+img_set+"/images"
    gtfilepath=rootdir+"/wider_face_split/wider_face_"+img_set+"_bbx_gt.txt"
    imagesdir=rootdir+"/images"
    vocannotationdir=rootdir+"/Annotations"
    labelsdir=rootdir+"/labels"
    if not os.path.exists(imagesdir):
        os.mkdir(imagesdir)
    if convet2yoloformat:
        if not os.path.exists(labelsdir):
            os.mkdir(labelsdir)
    if convert2vocformat:
        if not os.path.exists(vocannotationdir):
            os.mkdir(vocannotationdir)
    index=0
    with open(gtfilepath,'r') as gtfile:
        while(True ):#and len(faces)<10
            filename=gtfile.readline()[:-1]
            if(filename==""):
                break;
            sys.stdout.write("\r"+str(index)+":"+filename+"\t\t\t")
            sys.stdout.flush()
            imgpath=imgdir+"/"+filename
            img=cv2.imread(imgpath)
            if not img.data:
                break;
            imgheight=img.shape[0]
            imgwidth=img.shape[1]
            maxl=max(imgheight,imgwidth)
            paddingleft=(maxl-imgwidth)>>1
            paddingright=(maxl-imgwidth)>>1
            paddingbottom=(maxl-imgheight)>>1
            paddingtop=(maxl-imgheight)>>1
            saveimg=cv2.copyMakeBorder(img,paddingtop,paddingbottom,paddingleft,paddingright,cv2.BORDER_CONSTANT,value=0)
            showimg=saveimg.copy()
            numbbox=int(gtfile.readline())
            bboxes=[]
            for i in range(numbbox):
                line=gtfile.readline()
                line=line.split()
                line=line[0:4]               
                if(int(line[3])<=0 or int(line[2])<=0):
                    continue
                x=int(line[0])+paddingleft
                y=int(line[1])+paddingtop
                width=int(line[2])
                height=int(line[3])
                bbox=(x,y,width,height)
                x2=x+width
                y2=y+height
                #face=img[x:x2,y:y2]
                if width>=minsize2select and height>=minsize2select:
                    bboxes.append(bbox)
                    cv2.rectangle(showimg,(x,y),(x2,y2),(0,255,0))
                    #maxl=max(width,height)
                    #x3=(int)(x+(width-maxl)*0.5)
                    #y3=(int)(y+(height-maxl)*0.5)
                    #x4=(int)(x3+maxl)
                    #y4=(int)(y3+maxl)
                    #cv2.rectangle(img,(x3,y3),(x4,y4),(255,0,0))
                else:
                    cv2.rectangle(showimg,(x,y),(x2,y2),(0,0,255))              
            filename=filename.replace("/","_")
            if len(bboxes)==0:
                print("warrning: no face")
                continue 
            cv2.imwrite(imagesdir+"/"+filename,saveimg)
            if convet2yoloformat:
                height=saveimg.shape[0]
                width=saveimg.shape[1]
                txtpath=labelsdir+"/"+filename
                txtpath=txtpath[:-3]+"txt"
                ftxt=open(txtpath,'w')  
                for i in range(len(bboxes)):
                    bbox=bboxes[i]
                    xcenter=(bbox[0]+bbox[2]*0.5)/width
                    ycenter=(bbox[1]+bbox[3]*0.5)/height
                    wr=bbox[2]*1.0/width
                    hr=bbox[3]*1.0/height
                    txtline="0 "+str(xcenter)+" "+str(ycenter)+" "+str(wr)+" "+str(hr)+"\n"
                    ftxt.write(txtline)
                ftxt.close()
            if convert2vocformat:
                xmlpath=vocannotationdir+"/"+filename
                xmlpath=xmlpath[:-3]+"xml"
                doc = Document()
                annotation = doc.createElement('annotation')
                doc.appendChild(annotation)
                folder = doc.createElement('folder')
                folder_name = doc.createTextNode('widerface')
                folder.appendChild(folder_name)
                annotation.appendChild(folder)
                filenamenode = doc.createElement('filename')
                filename_name = doc.createTextNode(filename)
                filenamenode.appendChild(filename_name)
                annotation.appendChild(filenamenode)
                source = doc.createElement('source')
                annotation.appendChild(source)
                database = doc.createElement('database')
                database.appendChild(doc.createTextNode('wider face Database'))
                source.appendChild(database)
                annotation_s = doc.createElement('annotation')
                annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
                source.appendChild(annotation_s)
                image = doc.createElement('image')
                image.appendChild(doc.createTextNode('flickr'))
                source.appendChild(image)
                flickrid = doc.createElement('flickrid')
                flickrid.appendChild(doc.createTextNode('-1'))
                source.appendChild(flickrid)
                owner = doc.createElement('owner')
                annotation.appendChild(owner)
                flickrid_o = doc.createElement('flickrid')
                flickrid_o.appendChild(doc.createTextNode('tsaith'))
                owner.appendChild(flickrid_o)
                name_o = doc.createElement('name')
                name_o.appendChild(doc.createTextNode('tsaith'))
                owner.appendChild(name_o)
                size = doc.createElement('size')
                annotation.appendChild(size)
                width = doc.createElement('width')
                width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
                height = doc.createElement('height')
                height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
                depth = doc.createElement('depth')
                depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))
                size.appendChild(width)
                size.appendChild(height)
                size.appendChild(depth)
                segmented = doc.createElement('segmented')
                segmented.appendChild(doc.createTextNode('0'))
                annotation.appendChild(segmented)
                for i in range(len(bboxes)):
                    bbox=bboxes[i]
                    objects = doc.createElement('object')
                    annotation.appendChild(objects)
                    object_name = doc.createElement('name')
                    object_name.appendChild(doc.createTextNode('face'))
                    objects.appendChild(object_name)
                    pose = doc.createElement('pose')
                    pose.appendChild(doc.createTextNode('Unspecified'))
                    objects.appendChild(pose)
                    truncated = doc.createElement('truncated')
                    truncated.appendChild(doc.createTextNode('1'))
                    objects.appendChild(truncated)
                    difficult = doc.createElement('difficult')
                    difficult.appendChild(doc.createTextNode('0'))
                    objects.appendChild(difficult)
                    bndbox = doc.createElement('bndbox')
                    objects.appendChild(bndbox)
                    xmin = doc.createElement('xmin')
                    xmin.appendChild(doc.createTextNode(str(bbox[0])))
                    bndbox.appendChild(xmin)
                    ymin = doc.createElement('ymin')
                    ymin.appendChild(doc.createTextNode(str(bbox[1])))
                    bndbox.appendChild(ymin)
                    xmax = doc.createElement('xmax')
                    xmax.appendChild(doc.createTextNode(str(bbox[0]+bbox[2])))
                    bndbox.appendChild(xmax)
                    ymax = doc.createElement('ymax')
                    ymax.appendChild(doc.createTextNode(str(bbox[1]+bbox[3])))
                    bndbox.appendChild(ymax)
                f=open(xmlpath,"w")
                f.write(doc.toprettyxml(indent = ''))
                f.close()     
            #cv2.imshow("img",showimg)
            #cv2.waitKey()
            index=index+1
 
def generatetxt(img_set="train"):
    gtfilepath=rootdir+"/wider_face_split/wider_face_"+img_set+"_bbx_gt.txt"
    f=open(rootdir+"/"+img_set+".txt","w")
    with open(gtfilepath,'r') as gtfile:
        while(True ):#and len(faces)<10
            filename=gtfile.readline()[:-1]
            if(filename==""):
                break;
            filename=filename.replace("/","_")
            imgfilepath=datasetprefix+"/images/"+filename
            f.write(imgfilepath+'\n')
            numbbox=int(gtfile.readline())
            for i in range(numbbox):
                line=gtfile.readline()
    f.close()
 
def generatevocsets(img_set="train"):
    if not os.path.exists(rootdir+"/ImageSets"):
        os.mkdir(rootdir+"/ImageSets")
    if not os.path.exists(rootdir+"/ImageSets/Main"):
        os.mkdir(rootdir+"/ImageSets/Main")
    gtfilepath=rootdir+"/wider_face_split/wider_face_"+img_set+"_bbx_gt.txt"
    f=open(rootdir+"/ImageSets/Main/"+img_set+".txt",'w')
    with open(gtfilepath,'r') as gtfile:
        while(True ):#and len(faces)<10
            filename=gtfile.readline()[:-1]
            if(filename==""):
                break;
            filename=filename.replace("/","_")
            imgfilepath=filename[:-4]
            f.write(imgfilepath+'\n')
            numbbox=int(gtfile.readline())
            for i in range(numbbox):
                line=gtfile.readline()
    f.close()
 
def convertdataset():
    img_sets=["train","val"]
    for img_set in img_sets:
        convertimgset(img_set)
        generatetxt(img_set)
        generatevocsets(img_set)
 
if __name__=="__main__":
    convertdataset()
    shutil.move(rootdir+"/"+"train.txt",rootdir+"/"+"trainval.txt")
    shutil.move(rootdir+"/"+"val.txt",rootdir+"/"+"test.txt")
    shutil.move(rootdir+"/ImageSets/Main/"+"train.txt",rootdir+"/ImageSets/Main/"+"trainval.txt")
    shutil.move(rootdir+"/ImageSets/Main/"+"val.txt",rootdir+"/ImageSets/Main/"+"test.txt")
