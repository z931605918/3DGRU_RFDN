import os

def rename():
    path = 'D:\desktop\连续流场数据集\双圆柱绕流4000\\test2'
    save_name='lite'
    dirlist=os.listdir(path)
    i = 0
    for files in dirlist:
        if files.startswith('f'):
            dir=os.path.join(path,files)
            files_list=os.listdir(dir)

            for file_name in files_list:
                odd_dir=os.path.join(dir,file_name)
                filename = os.path.splitext(file_name)[0]
                filetype = os.path.splitext(file_name)[1]
                new_name = 'group_{:02d}_number_'.format(i)+str(filename[-4:])
                new_dir = os.path.join(path+'\\'+save_name,new_name+filetype )
                os.rename(odd_dir,new_dir)
            i+=1

def rename2():
    path = 'H:\image_enhancement\\trainingset2\label5'
    new_path='H:\image_enhancement\\trainingset2\labelp'
    save_name='p'
    dirlist=os.listdir(path)
    i = 0
    for files in dirlist:

        dir=os.path.join(path,files)
        files_list=os.listdir(path)

        for file_name in files_list:
            odd_dir=os.path.join(dir,file_name)
            filename = os.path.splitext(file_name)[0]
            filetype = os.path.splitext(file_name)[1]
            new_name = 'p'+str(filename[-4:])




            new_dir=os.path.join(new_path,new_name+filetype)
            os.rename(odd_dir,new_dir)
        i+=1


rename2()