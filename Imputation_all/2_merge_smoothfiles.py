# -*- coding: utf-8 -*-
"""
@author: Davide Aureli, Renato Bruni, Cinzia Daraio

"""

# =============================================================================
# LIBRARY
# =============================================================================

import shutil
import glob
import pandas as pd
from tqdm import tqdm

# =============================================================================
# PARAMETERS - Defined here, the user can change them
# =============================================================================

#HERE Flag about the Imputation (After_Smooth or After_Donor)
flag_time = "after_smooth"
#flag_time = "after_donor"

#Read Research_analysis parameter
with open('research_analysis.txt', encoding='utf-8') as f:
    research_analysis = f.read().replace('\n', '')


# Read back "path.txt", called in the Smooth Code
if flag_time == "after_smooth":
    
    #Path smooth files
    with open('path_smooth.txt', encoding='utf-8') as f:
        path = f.read().replace('\n', '')

elif flag_time == "after_donor":
    
    #Path smooth files
    with open('path_donor.txt', encoding='utf-8') as f:
        path = f.read().replace('\n', '')
    
    #Path starting file    
    with open('path_imputation_donor.txt', encoding='utf-8') as f:
        path_start = f.read().replace('\n', '')


#All Excel Files created during imputation with Smooth or Donor
files_Excel = sorted(glob.glob(path + '/*xlsx'))

#List of all Files found
print(files_Excel)

#The original dataset to take the columns name
if flag_time == "after_smooth":
    name_starting_file = ".\original_dataset.xlsx"

elif flag_time == "after_donor":
    name_starting_file = path_start + "fileout_donor_relaxed.xlsx"

#Final Excel File name
if flag_time == "after_smooth":
    final_name_file = path + "/fileout_smooth_complete.xlsx"
    
elif flag_time == "after_donor":
    final_name_file = path_start + "/fileout_donor_complete.xlsx"



#Creation of the Final DataSmooth File
shutil.copy(name_starting_file,final_name_file)

#Reading the final dataset
DataSmooth = pd.read_excel(final_name_file)

#Usefull for the imputation after donor
interesting_coloumns = DataSmooth.columns

#Mapping Columns between variable imputed and col_names in the final dataset

if flag_time == "after_smooth":
    
    column_names = {
     "fileout_students":"Smooth Students Enrolled",
     "fileout_graduates":"Smooth Students Graduates",
     "fileout_phd students":"Smooth PhD Enrolled",
     "fileout_phd graduates":"Smooth PhD Graduates",
     "fileout_academic staff FTE":"Smooth Academic Staff FTE",
     "fileout_academic staff HC":"Smooth Academic Staff HC",
     "fileout_non academic staff FTE":"Smooth Non Academic Staff FTE",
     "fileout_non academic staff HC":"Smooth Non Academic Staff HC",
     "fileout_expenditure":"Smooth Expenditure (EURO)",
     "fileout_revenues":"Smooth Revenues (EURO)"}

elif flag_time == "after_donor":

    column_names = {
     "fileout_students":"Value Donor Enrolled",
     "fileout_graduates":"Value Donor Graduates",
     "fileout_phd students":"Value Donor PhD Enrolled",
     "fileout_phd graduates":"Smooth PhD Graduates",
     "fileout_academic staff FTE":"Value Donor Academic Staff (FTE)",
     "fileout_academic staff HC":"Value Donor Academic Staff (HC)",
     "fileout_non academic staff FTE":"Value Donor Non Academic Staff (FTE)",
     "fileout_non academic staff HC":"Value Donor Non Academic Staff (HC)",
     "fileout_expenditure":"Value Donor Expenditure (EURO)",
     "fileout_revenues":"Value Donor Revenues (EURO)"}


#Make a cycle with the ordered name of Smooth Variables according to the imputation order
#So the first variable imputed will be Total Students Enrolled at ISCED 5-7 until Revenues(EURO)

for i in tqdm(range(len(files_Excel))):
    
    if files_Excel[i].split("\\")[-1].split(".")[0] in column_names:
        
        #print("Take data from: " + str(files_Excel[i]))
        
        #We read the Excel created just at the end of the first imputation
        data_smooth = pd.read_excel(files_Excel[i])
        #Extract the column Imputed
        column_add = data_smooth["Imputation"]
        
        #Add Column to the final dataset
        DataSmooth[column_names[files_Excel[i].split("\\")[-1].split(".")[0]]] = column_add




if flag_time == "after_smooth":
    
    #Check about research_analysis declared by the user
    
    if research_analysis == "1":
        var_ordered = ['ETER ID', 'Reference year', 'Institution Name',
               'Institution Category standardized', 'Institution Category - English',
               'English Institution Name', 'Country Code',
               'Distance education institution', 'Legal status',
                      'Total students enrolled ISCED 5-7', 'Smooth Students Enrolled',
               'Total graduates ISCED 5-7', 'Smooth Students Graduates',
               'Total students enrolled at ISCED 8', 'Smooth PhD Enrolled',
               'Total graduates at ISCED 8', 'Smooth PhD Graduates',
               'Total academic staff (FTE)', 'Smooth Academic Staff FTE',
               'Total academic staff (HC)', 'Smooth Academic Staff HC',
               'Number of non-academic  staff (FTE)', 'Smooth Non Academic Staff FTE',
               'Number of non-academic staff (HC)', 'Smooth Non Academic Staff HC',
               'Total Current expenditure (EURO)', 'Smooth Expenditure (EURO)',
               'Total Current revenues (EURO)', 'Smooth Revenues (EURO)',
        
               'Research active institution', 'Reasearch Active Imputed',
               'FLAG Reasearch Active', 'p', 'pp(top 10)', 'mcs', 'mncs',
               'pp(industry)', 'pp(int collab)', 'pp(collab)',
               'Lowest degree delivered', 'Highest degree delivered']
        
    elif research_analysis == "0":
        
        var_ordered = ['ETER ID', 'Reference year', 'Institution Name',
               'Institution Category standardized', 'Institution Category - English',
               'English Institution Name', 'Country Code',
               'Distance education institution', 'Legal status',
                      'Total students enrolled ISCED 5-7', 'Smooth Students Enrolled',
               'Total graduates ISCED 5-7', 'Smooth Students Graduates',
               'Total students enrolled at ISCED 8', 'Smooth PhD Enrolled',
               'Total graduates at ISCED 8', 'Smooth PhD Graduates',
               'Total academic staff (FTE)', 'Smooth Academic Staff FTE',
               'Total academic staff (HC)', 'Smooth Academic Staff HC',
               'Number of non-academic  staff (FTE)', 'Smooth Non Academic Staff FTE',
               'Number of non-academic staff (HC)', 'Smooth Non Academic Staff HC',
               'Total Current expenditure (EURO)', 'Smooth Expenditure (EURO)',
               'Total Current revenues (EURO)', 'Smooth Revenues (EURO)']        

elif flag_time == "after_donor":
    
    #interesting_columns defind above
    var_ordered = interesting_coloumns 



DataSmooth = DataSmooth[var_ordered]


DataSmooth.to_excel(final_name_file, sheet_name='Sheet_name_1',index = False)



# =============================================================================
# POSTPROCESSING
# =============================================================================

#Postprocessing to improve data Ratios, the user could modify this part if needed
#but he must know what he is doing. There is no easy way to change it in a black box 
#style.

#This part considers the final correlation between the imputed variables
starting_smooth = pd.read_excel(final_name_file)


#In this part we analyse all the values imputed according to the correlation between variables, 
#observing if the ratio between the 2 variables is still included within the Fork ([min,max]).
#In case we find a value which does not respect this constraint, it will be put back, or at the
#min edge or at the max edge.

#Upper Bound and Lower Bound 

max_ub_fork = 0.3
min_lb_fork = 0.2

def postProcessing(dataset,var_smooth_1,var_smooth_2,var_numerator,var_denominator):
    
    
    #Code for the Correlation and Ratio between the 2 variables specified above.
    starting_smooth[var_smooth_1] = starting_smooth[var_smooth_1].replace(["x", "xc", "xr", "c", "s","nc"], "m" )
    starting_smooth[var_smooth_2] = starting_smooth[var_smooth_2].replace(["x", "xc", "xr", "c", "s","nc"], "m" )
    
    #All possible ETER ID
    for i in set(dataset["ETER ID"]):
        
        #Extract the dataset related to the specific ETER ID
        data_lavoro = dataset[dataset["ETER ID"] == i].copy()
        
        #"Fork" or "Window" values for our analysis
        forchetta = []
        for anni in set(data_lavoro["Reference year"]):
            
            #Denominator
            den = data_lavoro[data_lavoro["Reference year"] == anni][var_denominator].values[0]
            
            #Numerator
            num = data_lavoro[data_lavoro["Reference year"] == anni][var_numerator].values[0]
            
            try:
                forchetta.append(num/den)
            except:
                pass
    
        if len(forchetta) >= 1:
    
            #print(Fork)
            #print()
            #print(max(forchetta)*0.3 + max(forchetta))
            #print(min(forchetta) - 0.2*(min(forchetta)))
            
            massimo = max(forchetta)*max_ub_fork + max(forchetta)
            minimo = min(forchetta) - min_lb_fork*(min(forchetta)) 
            
            #print("Index")
            indice = list(data_lavoro.index)
            #print(list(data_lavoro.index))
            
            for ind in indice:
                #print()
                #print(ind)
                
                num_2 = data_lavoro.loc[ind][var_smooth_2]
                den_2 = data_lavoro.loc[ind][var_smooth_1]
                
                try:
                    valore_check = num_2/den_2
                    
                    #print()
                    #print("Numerator")
                    #print(num_2)
                    #print("Denominator")
                    #print(den_2)
                    #print()
                    
                    if valore_check >= massimo:
                        
                        #print(valore_check)
                        #print(massimo)
    
                        val_nuovo = den_2*massimo
                        
                        dataset[var_smooth_2][ind] = round(val_nuovo,0)
    
                    elif valore_check<= minimo:
    
                        
                        #print(valore_check)
                        #print(minimo)
    
                        val_nuovo = den_2*minimo
                        
                        dataset[var_smooth_2][ind] = round(val_nuovo,0)
                except:
                    pass

    return dataset




if flag_time == "after_smooth":
    
    starting_smooth =  postProcessing(starting_smooth,'Total students enrolled ISCED 5-7',
                                      'Total graduates ISCED 5-7','Smooth Students Graduates',
                                      'Smooth Students Enrolled' )
    
    starting_smooth =  postProcessing(starting_smooth,'Total students enrolled at ISCED 8',
                                      'Total graduates at ISCED 8','Smooth PhD Graduates',
                                      'Smooth PhD Enrolled' )

elif flag_time == "after_donor":
    
    starting_smooth =  postProcessing(starting_smooth,'Smooth Students Enrolled',
                                      'Smooth Students Graduates','Value Donor Graduates',
                                      'Value Donor Enrolled' )
    
    starting_smooth =  postProcessing(starting_smooth,'Smooth PhD Enrolled',
                                      'Total graduates at ISCED 8','Value Donor PhD Graduates',
                                      'Value Donor PhD Enrolled' )



#Save the final File Excel
starting_smooth.to_excel(final_name_file, sheet_name='Sheet_name_1', index = False)

print("Merge Smooth Files Completed")
print() 