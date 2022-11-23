#Utility functions file 

def read_data(f):

    dict_images={}
    k=0

    lines=[line for line in open(f)]

    for i in range(0,len(lines),28):
        dict_images[k]=lines[i:i+28]
        k+=1

    return dict_images

traindata='./digitdata/trainingimages'
read_data(traindata)