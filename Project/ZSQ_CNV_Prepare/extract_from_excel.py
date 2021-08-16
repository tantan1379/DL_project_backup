import xlrd
import os
import shutil


table_path = './filter.xlsx'
des_path = 'F:/Dataset/ZSQ_selected'


data = xlrd.open_workbook(table_path)
table = data.sheets()[0]

index_list = []
name_list = []
oct_1_list = []
oct_2_list = []
oct_3_list = []
oct_4_list = []

# get list from table
for index in table.col_values(0, start_rowx=1, end_rowx=75):
    index_list.append(index)
    
for name in table.col_values(1, start_rowx=1, end_rowx=75):
    name_list.append(name)
    
for oct1 in table.col_values(2, start_rowx=1, end_rowx=75):
    oct_1_list.append(oct1)
for oct2 in table.col_values(3, start_rowx=1, end_rowx=75):
    oct_2_list.append(oct2)
for oct3 in table.col_values(4, start_rowx=1, end_rowx=75):
    oct_3_list.append(oct3)
for oct4 in table.col_values(5, start_rowx=1, end_rowx=75):
    oct_4_list.append(oct4)

# make files
for index in index_list:
    if not os.path.exists(os.path.join(des_path,index)):
        os.mkdir(os.path.join(des_path,index))
    if not os.path.exists(os.path.join(des_path,index,'oct1')):
        os.mkdir(os.path.join(des_path,index,'oct1'))
    if not os.path.exists(os.path.join(des_path,index,'oct2')):
        os.mkdir(os.path.join(des_path,index,'oct2'))
    if not os.path.exists(os.path.join(des_path,index,'oct3')):
        os.mkdir(os.path.join(des_path,index,'oct3'))
    if not os.path.exists(os.path.join(des_path,index,'oct4')):
        os.mkdir(os.path.join(des_path,index,'oct4'))

# copy
for i in range(len(index_list)):
    shutil.copy(oct_1_list[i],os.path.join(des_path,index_list[i],'oct1'))
    shutil.copy(oct_2_list[i],os.path.join(des_path,index_list[i],'oct2'))
    shutil.copy(oct_3_list[i],os.path.join(des_path,index_list[i],'oct3'))
    shutil.copy(oct_4_list[i],os.path.join(des_path,index_list[i],'oct4'))