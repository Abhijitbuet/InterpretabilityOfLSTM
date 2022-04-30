import pandas as pd
import glob
import os

path = 'data/usgs_streamflow/' # use your path
all_files = []
for i in range(1,19):
    data = ""
    if i < 10:
        data = "0"+str(i)
    else:
        data = str(i)
    new_path = path+data+"/"
    #print(new_path)
    #print(glob.glob(new_path+"/*"))
    list_of_files = glob.glob(new_path+ "/*.txt")
    for file in list_of_files:
        all_files.append(file)
    #print(len(all_files))
print(len(all_files))
li = []

for filename in all_files:
    #print(filename)
    df = pd.read_csv(filename, header=None, delim_whitespace=True,on_bad_lines='skip')
    #print(filename)

    df =df.drop(df.columns[5], axis=1)
    #print(df)
    df.columns = [ 'gauge_id','year', 'month', 'day', 'observed_flow']

    df = df.loc[df['year'].isin([1980])]

    #print(df)
    if len(df.columns)!=5:
        print("error!")
    li.append(df)


dataframe = pd.concat(li, axis=0, ignore_index=True)
#print(dataframe)
data = pd.read_csv('data/camels_chars.txt', index_col='gauge_id')
ids = pd.read_csv('data/basin_list.txt', header=None, index_col=0)
selected_metadata = data[data.index.isin(ids.index)]
#print(ids.index)
#print(dataframe['gauge_id'])
selected_dataFrame = dataframe.loc[dataframe['gauge_id'].isin(ids.index)]

#print(selected_metadata)
#print(selected_dataFrame)
for column in selected_metadata.columns:
    #print(selected_metadata[column])
    print(column)
    selected_dataFrame[column] = selected_dataFrame['gauge_id'].map(selected_metadata[column])
    #pass

#print((selected_dataFrame))
selected_dataFrame.to_csv( 'data/interpreter_data.csv',index=False)