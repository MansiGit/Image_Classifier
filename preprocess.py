traindata='/content/drive/MyDrive/Final Project/digitdata/testimages'

with open(traindata) as f:
    lines = f.readlines()

dict_train={k:None for k in range(4000)}
k=0
arr=[]

for i in range(len(lines)):
  arr=[]
  while("#" in lines[i] or "+" in lines[i]):
    arr.append(lines[i]) 
    i+=1
  if not("#" in lines[i] or "+" in lines[i]):
    continue
  dict_train[k]=arr
  print(arr)
  k+=1

print(dict_train)