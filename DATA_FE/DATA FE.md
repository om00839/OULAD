
# YBIGTA CONFERENCE _ OPEN EDU


```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
%matplotlib inline
```


```python
pd.set_option('display.max_columns', 500)
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 1000)
```


```python
# original
assessments=pd.read_csv('assessments.csv')
courses=pd.read_csv('courses.csv')
studentassessment=pd.read_csv('studentAssessment.csv')
studentinfo=pd.read_csv('studentInfo.csv')
studentregistration=pd.read_csv('studentRegistration.csv')
studentVle=pd.read_csv('studentVle.csv')
vle=pd.read_csv('vle.csv')
```


```python
student_total = pd.merge(studentinfo, studentregistration,how='inner', on=['code_module', 'code_presentation', 'id_student'])
```


```python
assessment_total = pd.merge(studentassessment, assessments, on=['id_assessment','assessment_type'], how='left')
```

## assessment_total


```python
# Sort assessment_total
assessment_total = assessment_total.sort_values(['code_module', 'code_presentation', 'id_student', 'date'])
assessment_total = assessment_total[['code_module', 'code_presentation', 'id_student',
                                     'date', 'id_assessment', 'date_submitted', 'is_banked', 'score', 'assessment_type', 'weight']]
assessment_total.reset_index(drop = True, inplace=True)
```


```python
# Make new columns for other assessments
assessment_total.columns=['code_module', 'code_presentation', 'id_student', 
                          'date1','id_assessment1', 'date_submitted1', 'is_banked1', 'score1','assessment_type1', 'weight1']

assessment_total=assessment_total.reindex(columns=['code_module', 'code_presentation', 'id_student',
                                  'date1','id_assessment1', 'date_submitted1', 'is_banked1', 'score1','assessment_type1', 'weight1',
                                  'date2','id_assessment2', 'date_submitted2', 'is_banked2', 'score2','assessment_type2', 'weight2',
                                  'date3','id_assessment3', 'date_submitted3', 'is_banked3', 'score3','assessment_type3', 'weight3',
                                  'date4','id_assessment4', 'date_submitted4', 'is_banked4', 'score4','assessment_type4', 'weight4',
                                  'date5','id_assessment5', 'date_submitted5', 'is_banked5', 'score5','assessment_type5', 'weight5',
                                  'date6','id_assessment6', 'date_submitted6', 'is_banked6', 'score6','assessment_type6', 'weight6',
                                  'date7','id_assessment7', 'date_submitted7', 'is_banked7', 'score7','assessment_type7', 'weight7',
                                  'date8','id_assessment8', 'date_submitted8', 'is_banked8', 'score8','assessment_type8', 'weight8',
                                  'date9','id_assessment9', 'date_submitted9', 'is_banked9', 'score9','assessment_type9', 'weight9',
                                  'date10','id_assessment10', 'date_submitted10', 'is_banked10', 'score10','assessment_type10', 'weight10',
                                  'date11','id_assessment11', 'date_submitted11', 'is_banked11', 'score11','assessment_type11', 'weight11',
                                  'date12','id_assessment12', 'date_submitted12', 'is_banked12', 'score12','assessment_type12', 'weight12',
                                  'date13','id_assessment13', 'date_submitted13', 'is_banked13', 'score13','assessment_type13', 'weight13',
                                  'date14','id_assessment14', 'date_submitted14', 'is_banked14', 'score14','assessment_type14', 'weight14',
                                 ])
```


```python
# Split by assessment_total code_module & code_presentation
assessment_total1=assessment_total[(assessment_total['code_module']=='AAA') & (assessment_total['code_presentation']=='2013J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total2=assessment_total[(assessment_total['code_module']=='AAA') & (assessment_total['code_presentation']=='2014J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total3=assessment_total[(assessment_total['code_module']=='BBB') & (assessment_total['code_presentation']=='2013B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total4=assessment_total[(assessment_total['code_module']=='BBB') & (assessment_total['code_presentation']=='2013J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total5=assessment_total[(assessment_total['code_module']=='BBB') & (assessment_total['code_presentation']=='2014B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total6=assessment_total[(assessment_total['code_module']=='BBB') & (assessment_total['code_presentation']=='2014J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total7=assessment_total[(assessment_total['code_module']=='CCC') & (assessment_total['code_presentation']=='2014B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total8=assessment_total[(assessment_total['code_module']=='CCC') & (assessment_total['code_presentation']=='2014J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total9=assessment_total[(assessment_total['code_module']=='DDD') & (assessment_total['code_presentation']=='2013B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total10=assessment_total[(assessment_total['code_module']=='DDD') & (assessment_total['code_presentation']=='2013J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total11=assessment_total[(assessment_total['code_module']=='DDD') & (assessment_total['code_presentation']=='2014B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total12=assessment_total[(assessment_total['code_module']=='DDD') & (assessment_total['code_presentation']=='2014J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total13=assessment_total[(assessment_total['code_module']=='EEE') & (assessment_total['code_presentation']=='2013J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total14=assessment_total[(assessment_total['code_module']=='EEE') & (assessment_total['code_presentation']=='2014B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total15=assessment_total[(assessment_total['code_module']=='EEE') & (assessment_total['code_presentation']=='2014J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total16=assessment_total[(assessment_total['code_module']=='FFF') & (assessment_total['code_presentation']=='2013B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total17=assessment_total[(assessment_total['code_module']=='FFF') & (assessment_total['code_presentation']=='2013J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total18=assessment_total[(assessment_total['code_module']=='FFF') & (assessment_total['code_presentation']=='2014B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total19=assessment_total[(assessment_total['code_module']=='FFF') & (assessment_total['code_presentation']=='2014J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total20=assessment_total[(assessment_total['code_module']=='GGG') & (assessment_total['code_presentation']=='2013J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total21=assessment_total[(assessment_total['code_module']=='GGG') & (assessment_total['code_presentation']=='2014B')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
assessment_total22=assessment_total[(assessment_total['code_module']=='GGG') & (assessment_total['code_presentation']=='2014J')].sort_values(by=['id_student', 'date1']).reset_index(drop=True)
```


```python
# Make into one line based on id_student 1
for i,k in zip(np.unique(assessment_total1['id_student'],return_index=True)[1], assessment_total1['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total1.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total1.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 2
for i,k in zip(np.unique(assessment_total2['id_student'],return_index=True)[1], assessment_total2['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total2.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total2.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 3
for i,k in zip(np.unique(assessment_total3['id_student'],return_index=True)[1], assessment_total3['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total3.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total3.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 4
for i,k in zip(np.unique(assessment_total4['id_student'],return_index=True)[1], assessment_total4['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total4.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total4.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 5
for i,k in zip(np.unique(assessment_total5['id_student'],return_index=True)[1], assessment_total5['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total5.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total5.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 6
for i,k in zip(np.unique(assessment_total6['id_student'],return_index=True)[1], assessment_total6['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total6.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total6.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 7
for i,k in zip(np.unique(assessment_total7['id_student'],return_index=True)[1], assessment_total7['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total7.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total7.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 8
for i,k in zip(np.unique(assessment_total8['id_student'],return_index=True)[1], assessment_total8['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total8.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total8.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 9
for i,k in zip(np.unique(assessment_total9['id_student'],return_index=True)[1], assessment_total9['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total9.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total9.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 10
for i,k in zip(np.unique(assessment_total10['id_student'],return_index=True)[1], assessment_total10['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total10.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total10.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 11
for i,k in zip(np.unique(assessment_total11['id_student'],return_index=True)[1], assessment_total11['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total11.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total11.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 12
for i,k in zip(np.unique(assessment_total12['id_student'],return_index=True)[1], assessment_total12['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total12.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total12.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 13
for i,k in zip(np.unique(assessment_total13['id_student'],return_index=True)[1], assessment_total13['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total13.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total13.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 14
for i,k in zip(np.unique(assessment_total14['id_student'],return_index=True)[1], assessment_total14['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total14.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total14.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 15
for i,k in zip(np.unique(assessment_total15['id_student'],return_index=True)[1], assessment_total15['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total15.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total15.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 16
for i,k in zip(np.unique(assessment_total16['id_student'],return_index=True)[1], assessment_total16['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total16.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total16.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 17
for i,k in zip(np.unique(assessment_total17['id_student'],return_index=True)[1], assessment_total17['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total17.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total17.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 18
for i,k in zip(np.unique(assessment_total18['id_student'],return_index=True)[1], assessment_total18['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total18.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total18.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 19
for i,k in zip(np.unique(assessment_total19['id_student'],return_index=True)[1], assessment_total19['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total19.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total19.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 20
for i,k in zip(np.unique(assessment_total20['id_student'],return_index=True)[1], assessment_total20['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total20.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total20.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 21
for i,k in zip(np.unique(assessment_total21['id_student'],return_index=True)[1], assessment_total21['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total21.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total21.iloc[i+j,3:10])
```


```python
# Make into one line based on id_student 22
for i,k in zip(np.unique(assessment_total22['id_student'],return_index=True)[1], assessment_total22['id_student'].value_counts().sort_index().values):
    for j in range(1,k):
        assessment_total22.iloc[i,10+7*(j-1):10+7*j]=np.array(assessment_total22.iloc[i+j,3:10])
```


```python
# row들을 다 정리하고 새로운 dataframe 'new1'~'new15'를 만들겠습니다.
new1=assessment_total1.iloc[np.unique(assessment_total1['id_student'],return_index=True)[1],:].reset_index(drop=True)
new2=assessment_total2.iloc[np.unique(assessment_total2['id_student'],return_index=True)[1],:].reset_index(drop=True)
new3=assessment_total3.iloc[np.unique(assessment_total3['id_student'],return_index=True)[1],:].reset_index(drop=True)
new4=assessment_total4.iloc[np.unique(assessment_total4['id_student'],return_index=True)[1],:].reset_index(drop=True)
new5=assessment_total5.iloc[np.unique(assessment_total5['id_student'],return_index=True)[1],:].reset_index(drop=True)
new6=assessment_total6.iloc[np.unique(assessment_total6['id_student'],return_index=True)[1],:].reset_index(drop=True)
new7=assessment_total7.iloc[np.unique(assessment_total7['id_student'],return_index=True)[1],:].reset_index(drop=True)
new8=assessment_total8.iloc[np.unique(assessment_total8['id_student'],return_index=True)[1],:].reset_index(drop=True)
new9=assessment_total9.iloc[np.unique(assessment_total9['id_student'],return_index=True)[1],:].reset_index(drop=True)
new10=assessment_total10.iloc[np.unique(assessment_total10['id_student'],return_index=True)[1],:].reset_index(drop=True)
new11=assessment_total11.iloc[np.unique(assessment_total11['id_student'],return_index=True)[1],:].reset_index(drop=True)
new12=assessment_total12.iloc[np.unique(assessment_total12['id_student'],return_index=True)[1],:].reset_index(drop=True)
new13=assessment_total13.iloc[np.unique(assessment_total13['id_student'],return_index=True)[1],:].reset_index(drop=True)
new14=assessment_total14.iloc[np.unique(assessment_total14['id_student'],return_index=True)[1],:].reset_index(drop=True)
new15=assessment_total15.iloc[np.unique(assessment_total15['id_student'],return_index=True)[1],:].reset_index(drop=True)
new16=assessment_total16.iloc[np.unique(assessment_total16['id_student'],return_index=True)[1],:].reset_index(drop=True)
new17=assessment_total17.iloc[np.unique(assessment_total17['id_student'],return_index=True)[1],:].reset_index(drop=True)
new18=assessment_total18.iloc[np.unique(assessment_total18['id_student'],return_index=True)[1],:].reset_index(drop=True)
new19=assessment_total19.iloc[np.unique(assessment_total19['id_student'],return_index=True)[1],:].reset_index(drop=True)
new20=assessment_total20.iloc[np.unique(assessment_total20['id_student'],return_index=True)[1],:].reset_index(drop=True)
new21=assessment_total21.iloc[np.unique(assessment_total21['id_student'],return_index=True)[1],:].reset_index(drop=True)
new22=assessment_total22.iloc[np.unique(assessment_total22['id_student'],return_index=True)[1],:].reset_index(drop=True)
```


```python
# 세로로 merge하기
assessment_total_new=pd.concat([new1,new2,new3,new4,new5,new6,new7,new8,new9,new10,new11,new12,new13,new14,new15,new16,new17,new18,new19,new20,new21,new22])
assessment_total_new.head()
```


```python
assessment_total_new=assessment_total_new.sort_values(by='id_student').reset_index(drop=True)
```


```python
assessment_total_new.to_csv('assessment_total_new.csv')
```

# The csv file above has a problem because the code simply concatenated information about assignment without making a space
### (Some students did not submit part of their assignments)
### (The problem-solved-version file was completed  manually through excel. This file is called)


```python
assessment_total_new_revised = pd.read_csv('assessment_total_new_revised.csv')
```

    C:\Users\hyunj\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2785: DtypeWarning: Columns (71,78,85,92,99) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
assessment_total_new_revised.sort_values(['code_module', 'code_presentation', 'id_student'], inplace=True)
assessment_total_new_revised.reset_index(drop=True, inplace=True)
```

## StudentVle data


```python
# courses에서 code_module_presentation 만들고 따로 가져오기
courses['code_module_presentation'] = courses['code_module'] + courses['code_presentation']
code_module_presentation = courses['code_module_presentation']
```


```python
# The last day of class
module_presentation_length = courses['module_presentation_length']
```


```python
assessments['code_module_presentation'] = assessments['code_module'] + assessments['code_presentation']
```

### module_presentation마다 각각의 stage 마지막 날짜 계산


```python
# 각 assessment 마감일자
ds_list = []
for cmp, mpl in zip(code_module_presentation, module_presentation_length):
    cond = (assessments['code_module_presentation'] == cmp) 
    dseries = assessments[cond].sort_values(by = ['date'])['date']
    ds_list.append(dseries)    
ds_list
```




    [0     19.0
     1     54.0
     2    117.0
     3    166.0
     4    215.0
     5      NaN
     Name: date, dtype: float64, 6      19.0
     7      54.0
     8     117.0
     9     166.0
     10    215.0
     11      NaN
     Name: date, dtype: float64, 24     19.0
     25     47.0
     31     54.0
     26     96.0
     32     96.0
     27    131.0
     33    131.0
     28    166.0
     34    166.0
     29    208.0
     35    208.0
     30      NaN
     Name: date, dtype: float64, 48     19.0
     49     54.0
     50    110.0
     51    152.0
     52    201.0
     53      NaN
     Name: date, dtype: float64, 12     19.0
     13     47.0
     19     54.0
     14     89.0
     20     89.0
     15    124.0
     21    124.0
     16    159.0
     22    159.0
     17    187.0
     23    187.0
     18      NaN
     Name: date, dtype: float64, 36     12.0
     37     40.0
     43     47.0
     38     82.0
     44     82.0
     39    117.0
     45    117.0
     40    152.0
     46    152.0
     41    194.0
     47    194.0
     42      NaN
     Name: date, dtype: float64, 68     18.0
     64     32.0
     69     67.0
     65    109.0
     70    144.0
     66    158.0
     67    207.0
     71    214.0
     72      NaN
     73      NaN
     Name: date, dtype: float64, 58     18.0
     54     32.0
     59     67.0
     55    102.0
     60    137.0
     56    151.0
     57    200.0
     61    207.0
     62      NaN
     63      NaN
     Name: date, dtype: float64, 88     25.0
     89     53.0
     90     88.0
     91    123.0
     92    165.0
     93    207.0
     94    261.0
     Name: date, dtype: float64, 102     20.0
     103     41.0
     104     62.0
     105    111.0
     106    146.0
     107    195.0
     108      NaN
     Name: date, dtype: float64, 81     23.0
     74     25.0
     82     51.0
     75     53.0
     83     79.0
     76     81.0
     84    114.0
     77    116.0
     85    149.0
     78    151.0
     86    170.0
     79    200.0
     87    206.0
     80    240.0
     Name: date, dtype: float64, 95      25.0
     96      53.0
     97      74.0
     98     116.0
     99     158.0
     100    200.0
     101    241.0
     Name: date, dtype: float64, 109     33.0
     110     68.0
     111    124.0
     112    159.0
     113    235.0
     Name: date, dtype: float64, 119     33.0
     120     68.0
     121    131.0
     122    166.0
     123    235.0
     Name: date, dtype: float64, 114     33.0
     115     68.0
     116    117.0
     117    152.0
     118    228.0
     Name: date, dtype: float64, 137     19.0
     138     47.0
     139     96.0
     140    131.0
     141    173.0
     142    236.0
     143    236.0
     144    236.0
     145    236.0
     146    236.0
     147    236.0
     148    236.0
     149    236.0
     Name: date, dtype: float64, 163     24.0
     164     52.0
     165     94.0
     166    136.0
     167    199.0
     168    241.0
     169    241.0
     170    241.0
     171    241.0
     172    241.0
     173    241.0
     174    241.0
     175    241.0
     Name: date, dtype: float64, 124     19.0
     125     47.0
     126     89.0
     127    131.0
     128    166.0
     129    222.0
     130    222.0
     131    222.0
     132    222.0
     133    222.0
     134    222.0
     135    222.0
     136    222.0
     Name: date, dtype: float64, 150     24.0
     151     52.0
     152     87.0
     153    129.0
     154    171.0
     155    227.0
     156    227.0
     157    227.0
     158    227.0
     159    227.0
     160    227.0
     161    227.0
     162    227.0
     Name: date, dtype: float64, 176     61.0
     177    124.0
     178    173.0
     179    229.0
     180    229.0
     181    229.0
     182    229.0
     183    229.0
     184    229.0
     185    229.0
     Name: date, dtype: float64, 196     61.0
     197    124.0
     198    173.0
     199    229.0
     200    229.0
     201    229.0
     202    229.0
     203    229.0
     204    229.0
     205    229.0
     Name: date, dtype: float64, 186     61.0
     187    117.0
     188    166.0
     189    222.0
     190    222.0
     191    222.0
     192    222.0
     193    222.0
     194    222.0
     195    222.0
     Name: date, dtype: float64]




```python
# nan값 -> module_presentation_length로 채움
filled_ds_list = []
for mpl, ds in zip(module_presentation_length, ds_list):
    filled_ds_list.append(ds.fillna(mpl))
filled_ds_list
```




    [0     19.0
     1     54.0
     2    117.0
     3    166.0
     4    215.0
     5    268.0
     Name: date, dtype: float64, 6      19.0
     7      54.0
     8     117.0
     9     166.0
     10    215.0
     11    269.0
     Name: date, dtype: float64, 24     19.0
     25     47.0
     31     54.0
     26     96.0
     32     96.0
     27    131.0
     33    131.0
     28    166.0
     34    166.0
     29    208.0
     35    208.0
     30    268.0
     Name: date, dtype: float64, 48     19.0
     49     54.0
     50    110.0
     51    152.0
     52    201.0
     53    262.0
     Name: date, dtype: float64, 12     19.0
     13     47.0
     19     54.0
     14     89.0
     20     89.0
     15    124.0
     21    124.0
     16    159.0
     22    159.0
     17    187.0
     23    187.0
     18    240.0
     Name: date, dtype: float64, 36     12.0
     37     40.0
     43     47.0
     38     82.0
     44     82.0
     39    117.0
     45    117.0
     40    152.0
     46    152.0
     41    194.0
     47    194.0
     42    234.0
     Name: date, dtype: float64, 68     18.0
     64     32.0
     69     67.0
     65    109.0
     70    144.0
     66    158.0
     67    207.0
     71    214.0
     72    269.0
     73    269.0
     Name: date, dtype: float64, 58     18.0
     54     32.0
     59     67.0
     55    102.0
     60    137.0
     56    151.0
     57    200.0
     61    207.0
     62    241.0
     63    241.0
     Name: date, dtype: float64, 88     25.0
     89     53.0
     90     88.0
     91    123.0
     92    165.0
     93    207.0
     94    261.0
     Name: date, dtype: float64, 102     20.0
     103     41.0
     104     62.0
     105    111.0
     106    146.0
     107    195.0
     108    262.0
     Name: date, dtype: float64, 81     23.0
     74     25.0
     82     51.0
     75     53.0
     83     79.0
     76     81.0
     84    114.0
     77    116.0
     85    149.0
     78    151.0
     86    170.0
     79    200.0
     87    206.0
     80    240.0
     Name: date, dtype: float64, 95      25.0
     96      53.0
     97      74.0
     98     116.0
     99     158.0
     100    200.0
     101    241.0
     Name: date, dtype: float64, 109     33.0
     110     68.0
     111    124.0
     112    159.0
     113    235.0
     Name: date, dtype: float64, 119     33.0
     120     68.0
     121    131.0
     122    166.0
     123    235.0
     Name: date, dtype: float64, 114     33.0
     115     68.0
     116    117.0
     117    152.0
     118    228.0
     Name: date, dtype: float64, 137     19.0
     138     47.0
     139     96.0
     140    131.0
     141    173.0
     142    236.0
     143    236.0
     144    236.0
     145    236.0
     146    236.0
     147    236.0
     148    236.0
     149    236.0
     Name: date, dtype: float64, 163     24.0
     164     52.0
     165     94.0
     166    136.0
     167    199.0
     168    241.0
     169    241.0
     170    241.0
     171    241.0
     172    241.0
     173    241.0
     174    241.0
     175    241.0
     Name: date, dtype: float64, 124     19.0
     125     47.0
     126     89.0
     127    131.0
     128    166.0
     129    222.0
     130    222.0
     131    222.0
     132    222.0
     133    222.0
     134    222.0
     135    222.0
     136    222.0
     Name: date, dtype: float64, 150     24.0
     151     52.0
     152     87.0
     153    129.0
     154    171.0
     155    227.0
     156    227.0
     157    227.0
     158    227.0
     159    227.0
     160    227.0
     161    227.0
     162    227.0
     Name: date, dtype: float64, 176     61.0
     177    124.0
     178    173.0
     179    229.0
     180    229.0
     181    229.0
     182    229.0
     183    229.0
     184    229.0
     185    229.0
     Name: date, dtype: float64, 196     61.0
     197    124.0
     198    173.0
     199    229.0
     200    229.0
     201    229.0
     202    229.0
     203    229.0
     204    229.0
     205    229.0
     Name: date, dtype: float64, 186     61.0
     187    117.0
     188    166.0
     189    222.0
     190    222.0
     191    222.0
     192    222.0
     193    222.0
     194    222.0
     195    222.0
     Name: date, dtype: float64]




```python
# module_presentation id 부여
studentVle['code_module_presentation'] = studentVle['code_module'] + studentVle['code_presentation']
```

### Add column 'stage' to studentVle


```python
# 굉장히 오래걸림

studentVle['stage'] = 0
for fds, cmp in zip(filled_ds_list, code_module_presentation) :
    
    tmp1 = fds.unique()
    
    last_assessment_date = tmp1[-1]
    last_course_date = courses.loc[(courses['code_module_presentation'] == cmp).values,'module_presentation_length'].values[0]
    print(last_assessment_date, last_course_date)
    
    for i in range(len(tmp1)):
        if i == 0 :
            idx = ((studentVle['code_module_presentation'] == cmp) & 
                   (studentVle['date'] <= tmp1[i]) &
                   (studentVle['date'] >= 0)).values
            studentVle.loc[idx,'stage'] = i+1

        else  :
            idx = ((studentVle['code_module_presentation'] == cmp) & 
                   (studentVle['date'] <= tmp1[i]) & 
                   (studentVle['date'] > tmp1[i-1])).values
            studentVle.loc[idx,'stage'] = i+1
                
    if last_assessment_date < last_course_date:

        idx = ((studentVle['code_module_presentation'] == cmp) & 
                       (studentVle['date'] <= last_course_date) & 
                       (studentVle['date'] > last_assessment_date)).values
        studentVle.loc[idx,'stage'] = len(tmp1)

studentVle
```


```python
# studentVle.to_csv('studentVle_stage.csv')
```

### Sum, Mean, Variance, nunique of sum_click during each stage


```python
studentVle_stage = pd.read_csv('studentVle_stage.csv', index_col=0)
```

    C:\Users\hyunj\Anaconda3\lib\site-packages\numpy\lib\arraysetops.py:472: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      mask |= (ar1 == a)



```python
sum_click_sum = studentVle_stage.groupby(['code_module', 'code_presentation', 'id_student', 'stage'])['sum_click'].sum().reset_index()
sum_click_sum.columns = ['code_module', 'code_presentation', 'id_student', 'stage', 'sum_click_sum']
```


```python
sum_click_mean = studentVle_stage.groupby(['code_module', 'code_presentation', 'id_student', 'stage'])['sum_click'].mean().reset_index()
sum_click_mean.columns = ['code_module', 'code_presentation', 'id_student', 'stage', 'sum_click_mean']
```


```python
sum_click_var = studentVle_stage.groupby(['code_module', 'code_presentation', 'id_student', 'stage'])['sum_click'].var().reset_index()
sum_click_var.columns = ['code_module', 'code_presentation', 'id_student', 'stage', 'sum_click_var']
```


```python
sum_click_nunique = studentVle_stage.groupby(['code_module', 'code_presentation', 'id_student', 'stage'])['date'].nunique().reset_index()
sum_click_nunique.columns = ['code_module', 'code_presentation', 'id_student', 'stage', 'sum_click_nunique']
```


```python
sum_click_data = pd.merge(sum_click_sum, sum_click_mean, on=['code_module','code_presentation','id_student','stage'], how='left')
sum_click_data = pd.merge(sum_click_data, sum_click_var, on=['code_module','code_presentation','id_student','stage'], how='left')
sum_click_data = pd.merge(sum_click_data, sum_click_nunique, on=['code_module','code_presentation','id_student','stage'], how='left')
```


```python
sum_click_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code_module</th>
      <th>code_presentation</th>
      <th>id_student</th>
      <th>stage</th>
      <th>sum_click_sum</th>
      <th>sum_click_mean</th>
      <th>sum_click_var</th>
      <th>sum_click_nunique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>11391</td>
      <td>0</td>
      <td>98</td>
      <td>8.909091</td>
      <td>160.690909</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>11391</td>
      <td>1</td>
      <td>303</td>
      <td>6.183673</td>
      <td>131.569728</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>11391</td>
      <td>2</td>
      <td>128</td>
      <td>4.266667</td>
      <td>30.685057</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>11391</td>
      <td>3</td>
      <td>99</td>
      <td>3.193548</td>
      <td>6.827957</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>11391</td>
      <td>4</td>
      <td>85</td>
      <td>6.071429</td>
      <td>63.763736</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## assessment_total_new_revised + sum_click_data


```python
values = sum_click_sum.pivot_table(index=['code_module', 'code_presentation', 'id_student'], columns=['stage']).reset_index().values
columns = ['code_module', 'code_presentation', 'id_student'] + ['sum_click_sum'+str(i) for i in range(15)]
temp = pd.DataFrame(data=values, columns=columns)
temp['id_student'] = temp['id_student'].astype(int)
assessment_total_new_revised = pd.merge(assessment_total_new_revised, temp, on = columns[:3], how = 'left')
```


```python
values = sum_click_mean.pivot_table(index=['code_module', 'code_presentation', 'id_student'], columns=['stage']).reset_index().values
columns = ['code_module', 'code_presentation', 'id_student'] + ['sum_click_mean'+str(i) for i in range(15)]
temp = pd.DataFrame(data=values, columns=columns)
temp['id_student'] = temp['id_student'].astype(int)
assessment_total_new_revised = pd.merge(assessment_total_new_revised, temp, on = columns[:3], how = 'left')
```


```python
values = sum_click_var.pivot_table(index=['code_module', 'code_presentation', 'id_student'], columns=['stage']).reset_index().values
columns = ['code_module', 'code_presentation', 'id_student'] + ['sum_click_var'+str(i) for i in range(15)]
temp = pd.DataFrame(data=values, columns=columns)
temp['id_student'] = temp['id_student'].astype(int)
assessment_total_new_revised = pd.merge(assessment_total_new_revised, temp, on = columns[:3], how = 'left')
```


```python
values = sum_click_nunique.pivot_table(index=['code_module', 'code_presentation', 'id_student'], columns=['stage']).reset_index().values
columns = ['code_module', 'code_presentation', 'id_student'] + ['sum_click_nunique'+str(i) for i in range(15)]
temp = pd.DataFrame(data=values, columns=columns)
temp['id_student'] = temp['id_student'].astype(int)
assessment_total_new_revised = pd.merge(assessment_total_new_revised, temp, on = columns[:3], how = 'left')
```

## student_total + assessment_total_new_revised


```python
final_df = pd.merge(student_total, assessment_total_new_revised, on=['code_module','code_presentation','id_student'], how='left')
final_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code_module</th>
      <th>code_presentation</th>
      <th>id_student</th>
      <th>gender</th>
      <th>region</th>
      <th>highest_education</th>
      <th>imd_band</th>
      <th>age_band</th>
      <th>num_of_prev_attempts</th>
      <th>studied_credits</th>
      <th>disability</th>
      <th>final_result</th>
      <th>date_registration</th>
      <th>date_unregistration</th>
      <th>date1</th>
      <th>id_assessment1</th>
      <th>date_submitted1</th>
      <th>is_banked1</th>
      <th>score1</th>
      <th>assessment_type1</th>
      <th>weight1</th>
      <th>date2</th>
      <th>id_assessment2</th>
      <th>date_submitted2</th>
      <th>is_banked2</th>
      <th>score2</th>
      <th>assessment_type2</th>
      <th>weight2</th>
      <th>date3</th>
      <th>id_assessment3</th>
      <th>date_submitted3</th>
      <th>is_banked3</th>
      <th>score3</th>
      <th>assessment_type3</th>
      <th>weight3</th>
      <th>date4</th>
      <th>id_assessment4</th>
      <th>date_submitted4</th>
      <th>is_banked4</th>
      <th>score4</th>
      <th>assessment_type4</th>
      <th>weight4</th>
      <th>date5</th>
      <th>id_assessment5</th>
      <th>date_submitted5</th>
      <th>is_banked5</th>
      <th>score5</th>
      <th>assessment_type5</th>
      <th>weight5</th>
      <th>date6</th>
      <th>id_assessment6</th>
      <th>date_submitted6</th>
      <th>is_banked6</th>
      <th>score6</th>
      <th>assessment_type6</th>
      <th>weight6</th>
      <th>date7</th>
      <th>id_assessment7</th>
      <th>date_submitted7</th>
      <th>is_banked7</th>
      <th>score7</th>
      <th>assessment_type7</th>
      <th>weight7</th>
      <th>date8</th>
      <th>id_assessment8</th>
      <th>date_submitted8</th>
      <th>is_banked8</th>
      <th>score8</th>
      <th>assessment_type8</th>
      <th>weight8</th>
      <th>date9</th>
      <th>id_assessment9</th>
      <th>date_submitted9</th>
      <th>is_banked9</th>
      <th>score9</th>
      <th>assessment_type9</th>
      <th>weight9</th>
      <th>date10</th>
      <th>id_assessment10</th>
      <th>date_submitted10</th>
      <th>is_banked10</th>
      <th>score10</th>
      <th>assessment_type10</th>
      <th>weight10</th>
      <th>date11</th>
      <th>id_assessment11</th>
      <th>date_submitted11</th>
      <th>is_banked11</th>
      <th>score11</th>
      <th>assessment_type11</th>
      <th>weight11</th>
      <th>date12</th>
      <th>id_assessment12</th>
      <th>date_submitted12</th>
      <th>is_banked12</th>
      <th>score12</th>
      <th>assessment_type12</th>
      <th>weight12</th>
      <th>date13</th>
      <th>id_assessment13</th>
      <th>date_submitted13</th>
      <th>is_banked13</th>
      <th>score13</th>
      <th>assessment_type13</th>
      <th>weight13</th>
      <th>date14</th>
      <th>id_assessment14</th>
      <th>date_submitted14</th>
      <th>is_banked14</th>
      <th>score14</th>
      <th>assessment_type14</th>
      <th>weight14</th>
      <th>sum_click_sum0</th>
      <th>sum_click_sum1</th>
      <th>sum_click_sum2</th>
      <th>sum_click_sum3</th>
      <th>sum_click_sum4</th>
      <th>sum_click_sum5</th>
      <th>sum_click_sum6</th>
      <th>sum_click_sum7</th>
      <th>sum_click_sum8</th>
      <th>sum_click_sum9</th>
      <th>sum_click_sum10</th>
      <th>sum_click_sum11</th>
      <th>sum_click_sum12</th>
      <th>sum_click_sum13</th>
      <th>sum_click_sum14</th>
      <th>sum_click_mean0</th>
      <th>sum_click_mean1</th>
      <th>sum_click_mean2</th>
      <th>sum_click_mean3</th>
      <th>sum_click_mean4</th>
      <th>sum_click_mean5</th>
      <th>sum_click_mean6</th>
      <th>sum_click_mean7</th>
      <th>sum_click_mean8</th>
      <th>sum_click_mean9</th>
      <th>sum_click_mean10</th>
      <th>sum_click_mean11</th>
      <th>sum_click_mean12</th>
      <th>sum_click_mean13</th>
      <th>sum_click_mean14</th>
      <th>sum_click_var0</th>
      <th>sum_click_var1</th>
      <th>sum_click_var2</th>
      <th>sum_click_var3</th>
      <th>sum_click_var4</th>
      <th>sum_click_var5</th>
      <th>sum_click_var6</th>
      <th>sum_click_var7</th>
      <th>sum_click_var8</th>
      <th>sum_click_var9</th>
      <th>sum_click_var10</th>
      <th>sum_click_var11</th>
      <th>sum_click_var12</th>
      <th>sum_click_var13</th>
      <th>sum_click_var14</th>
      <th>sum_click_nunique0</th>
      <th>sum_click_nunique1</th>
      <th>sum_click_nunique2</th>
      <th>sum_click_nunique3</th>
      <th>sum_click_nunique4</th>
      <th>sum_click_nunique5</th>
      <th>sum_click_nunique6</th>
      <th>sum_click_nunique7</th>
      <th>sum_click_nunique8</th>
      <th>sum_click_nunique9</th>
      <th>sum_click_nunique10</th>
      <th>sum_click_nunique11</th>
      <th>sum_click_nunique12</th>
      <th>sum_click_nunique13</th>
      <th>sum_click_nunique14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>11391</td>
      <td>M</td>
      <td>East Anglian Region</td>
      <td>HE Qualification</td>
      <td>90-100%</td>
      <td>55&lt;=</td>
      <td>0</td>
      <td>240</td>
      <td>N</td>
      <td>Pass</td>
      <td>-159.0</td>
      <td>NaN</td>
      <td>19.0</td>
      <td>1752.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>78.0</td>
      <td>TMA</td>
      <td>10.0</td>
      <td>54.0</td>
      <td>1753.0</td>
      <td>53.0</td>
      <td>0.0</td>
      <td>85.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>117.0</td>
      <td>1754.0</td>
      <td>115.0</td>
      <td>0.0</td>
      <td>80.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>166.0</td>
      <td>1755.0</td>
      <td>164.0</td>
      <td>0.0</td>
      <td>85.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>215.0</td>
      <td>1756.0</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>82.0</td>
      <td>TMA</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98</td>
      <td>303</td>
      <td>128</td>
      <td>99</td>
      <td>85</td>
      <td>78</td>
      <td>143</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.90909</td>
      <td>6.18367</td>
      <td>4.26667</td>
      <td>3.19355</td>
      <td>6.07143</td>
      <td>4.58824</td>
      <td>3.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>160.691</td>
      <td>131.57</td>
      <td>30.6851</td>
      <td>6.82796</td>
      <td>63.7637</td>
      <td>27.0074</td>
      <td>10.75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>7</td>
      <td>10</td>
      <td>8</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>28400</td>
      <td>F</td>
      <td>Scotland</td>
      <td>HE Qualification</td>
      <td>20-30%</td>
      <td>35-55</td>
      <td>0</td>
      <td>60</td>
      <td>N</td>
      <td>Pass</td>
      <td>-53.0</td>
      <td>NaN</td>
      <td>19.0</td>
      <td>1752.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>70.0</td>
      <td>TMA</td>
      <td>10.0</td>
      <td>54.0</td>
      <td>1753.0</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>68.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>117.0</td>
      <td>1754.0</td>
      <td>121.0</td>
      <td>0.0</td>
      <td>70.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>166.0</td>
      <td>1755.0</td>
      <td>164.0</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>215.0</td>
      <td>1756.0</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>TMA</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>215</td>
      <td>278</td>
      <td>176</td>
      <td>227</td>
      <td>172</td>
      <td>349</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.90909</td>
      <td>3.51899</td>
      <td>3.82609</td>
      <td>2.63953</td>
      <td>2.60606</td>
      <td>3.96591</td>
      <td>1.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.862</td>
      <td>10.5862</td>
      <td>14.458</td>
      <td>6.63324</td>
      <td>4.45781</td>
      <td>24.0333</td>
      <td>0.622222</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>8</td>
      <td>11</td>
      <td>19</td>
      <td>16</td>
      <td>15</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>30268</td>
      <td>F</td>
      <td>North Western Region</td>
      <td>A Level or Equivalent</td>
      <td>30-40%</td>
      <td>35-55</td>
      <td>0</td>
      <td>60</td>
      <td>Y</td>
      <td>Withdrawn</td>
      <td>-92.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>31604</td>
      <td>F</td>
      <td>South East Region</td>
      <td>A Level or Equivalent</td>
      <td>50-60%</td>
      <td>35-55</td>
      <td>0</td>
      <td>60</td>
      <td>N</td>
      <td>Pass</td>
      <td>-52.0</td>
      <td>NaN</td>
      <td>19.0</td>
      <td>1752.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>72.0</td>
      <td>TMA</td>
      <td>10.0</td>
      <td>54.0</td>
      <td>1753.0</td>
      <td>51.0</td>
      <td>0.0</td>
      <td>71.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>117.0</td>
      <td>1754.0</td>
      <td>115.0</td>
      <td>0.0</td>
      <td>74.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>166.0</td>
      <td>1755.0</td>
      <td>165.0</td>
      <td>0.0</td>
      <td>88.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>215.0</td>
      <td>1756.0</td>
      <td>213.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>TMA</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>169</td>
      <td>275</td>
      <td>360</td>
      <td>427</td>
      <td>426</td>
      <td>239</td>
      <td>262</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.44737</td>
      <td>3.35366</td>
      <td>2.88</td>
      <td>3.38889</td>
      <td>3.20301</td>
      <td>2.95062</td>
      <td>3.35897</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.362</td>
      <td>8.05857</td>
      <td>6.99355</td>
      <td>7.72756</td>
      <td>12.1479</td>
      <td>7.57253</td>
      <td>12.0513</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>11</td>
      <td>24</td>
      <td>28</td>
      <td>25</td>
      <td>12</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAA</td>
      <td>2013J</td>
      <td>32885</td>
      <td>F</td>
      <td>West Midlands Region</td>
      <td>Lower Than A Level</td>
      <td>50-60%</td>
      <td>0-35</td>
      <td>0</td>
      <td>60</td>
      <td>N</td>
      <td>Pass</td>
      <td>-176.0</td>
      <td>NaN</td>
      <td>19.0</td>
      <td>1752.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>TMA</td>
      <td>10.0</td>
      <td>54.0</td>
      <td>1753.0</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>117.0</td>
      <td>1754.0</td>
      <td>124.0</td>
      <td>0.0</td>
      <td>63.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>166.0</td>
      <td>1755.0</td>
      <td>181.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>TMA</td>
      <td>20.0</td>
      <td>215.0</td>
      <td>1756.0</td>
      <td>222.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>TMA</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>295</td>
      <td>204</td>
      <td>102</td>
      <td>146</td>
      <td>88</td>
      <td>78</td>
      <td>121</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.4697</td>
      <td>3.51724</td>
      <td>3.09091</td>
      <td>2.92</td>
      <td>2.14634</td>
      <td>1.69565</td>
      <td>2.08621</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.2221</td>
      <td>12.4295</td>
      <td>8.02273</td>
      <td>9.99347</td>
      <td>5.47805</td>
      <td>1.81643</td>
      <td>3.86963</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>11</td>
      <td>7</td>
      <td>14</td>
      <td>12</td>
      <td>7</td>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove rows that had unregistered before the class started
final_df=final_df.drop(final_df[final_df['date_unregistration']<=0].index).reset_index(drop=True)
```


```python
final_df.groupby('code_module')['id_student'].count()
```




    code_module
    AAA     728
    BBB    6921
    CCC    3935
    DDD    5676
    EEE    2701
    FFF    7069
    GGG    2466
    Name: id_student, dtype: int64



## Remove more rows that do not have assessment data through Excel (manually)


```python
total = pd.read_csv('total.csv', index_col=0, encoding='euc-kr')
```

    C:\Users\keris\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2785: DtypeWarning: Columns (69,76,83,90,97,104,111) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
# total.to_csv('total.csv')
```


```python

```
