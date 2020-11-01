# -*- coding: utf-8 -*-
"""
@author: Davide Aureli, Renato Bruni, Cinzia Daraio

"""

# =============================================================================
# LIBRARY
# =============================================================================

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle



# =============================================================================
# PARAMETERS - Defined here, the user can change them
# =============================================================================

#Research analysis
with open('research_analysis.txt', encoding='utf-8') as f:
    research_analysis = f.read().replace('\n', '')
    
#Read dataset after first part of Donor imputation
with open('path_imputation_donor.txt', encoding='utf-8') as f:
    path = f.read().replace('\n', '')
    
#directory and name of the saved file
name_file_start = "./fileout_donor_complete.xlsx"

name_file_final = "./imputed_dataset.xlsx"



# =============================================================================

#Dictionary variables
diz_variabili_colonne_modifiche = { 'Total students enrolled ISCED 5-7':["ID Donor Enrolled", 
                                                                         "Name Donor Enrolled", 
                                                                         "Distance Donor Enrolled",
                                                                         'Flag Enrolled',
                                                                         'Smooth Students Enrolled',
                                                                        "Value Donor Enrolled"],
                                   
                                   'Total graduates ISCED 5-7':["ID Donor Graduates",
                                                                "Name Donor Graduates", 
                                                                "Distance Donor Graduates",
                                                                "Flag Graduates",
                                                                'Smooth Students Graduates',
                                                               "Value Donor Graduates"],
                                   
                                   'Total students enrolled at ISCED 8':["ID Donor PhD Enrolled",
                                                                "Name Donor PhD Enrolled", 
                                                                "Distance Donor PhD Enrolled",
                                                                "Flag PhD Enrolled",
                                                                'Smooth PhD Enrolled',
                                                               "Value Donor PhD Enrolled"],
                                   
                                   'Total graduates at ISCED 8':["ID Donor PhD Graduates",
                                                                "Name Donor PhD Graduates", 
                                                                "Distance Donor PhD Graduates",
                                                                "Flag PhD Graduates",
                                                                'Smooth PhD Graduates',
                                                               "Value Donor PhD Graduates"],
                                   
                                   "Total academic staff (HC)":["ID Donor Academic Staff (HC)", 
                                                                "Name Donor Academic Staff (HC)", 
                                                             "Distance Donor Academic Staff (HC)" ,
                                                                "Flag Academic Staff (HC)",
                                                                'Smooth Academic Staff HC',
                                                              "Value Donor Academic Staff (HC)" ],
                                   
                                   
                                   
                                   'Total academic staff (FTE)':["ID Donor Academic Staff (FTE)", 
                                                                        "Name Donor Academic Staff (FTE)", 
                                                             "Distance Donor Academic Staff (FTE)" ,
                                                                 "Flag Academic Staff (FTE)",
                                                                  'Smooth Academic Staff FTE',
                                                                "Value Donor Academic Staff (FTE)"],
                           
                                    "Number of non-academic  staff (FTE)":["ID Donor Non Academic Staff (FTE)", 
                                                                        "Name Donor Non Academic Staff (FTE)", 
                                                             "Distance Donor Non Academic Staff (FTE)",
                                                                           "Flag Non Academic Staff (FTE)",
                                                                           'Smooth Non Academic Staff FTE',
                                                                          "Value Donor Non Academic Staff (FTE)"],
                           
                                    "Number of non-academic staff (HC)":["ID Donor Non Academic Staff (HC)", 
                                                                        "Name Donor Non Academic Staff (HC)", 
                                                             "Distance Donor Non Academic Staff (HC)",
                                                                         "Flag Non Academic Staff (HC)",
                                                                         'Smooth Non Academic Staff HC',
                                                                         "Value Donor Non Academic Staff (HC)"],
                                   
                                   'Total Current expenditure (EURO)':["ID Donor Expenditure (EURO)", 
                                                                       "Name Donor Expenditure (EURO)", 
                                                             "Distance Donor Expenditure (EURO)",
                                                                       "Flag Expenditure (EURO)",
                                                                       'Smooth Expenditure (EURO)',
                                                                      "Value Donor Expenditure (EURO)"],
                                   
                                   'Total Current revenues (EURO)':["ID Donor Revenues (EURO)",
                                                                    "Name Donor Revenues (EURO)", 
                                                                    "Distance Donor Revenues (EURO)",
                                                                    "Flag Revenues (EURO)",
                                                                    'Smooth Revenues (EURO)',
                                                                   "Value Donor Revenues (EURO)"]
                                    }


#Load the Dataset After the Smooth Imputation for the 2017 values.
#Add final results and Flag, updating the Ratio.


starting_dataset = pd.read_excel(path + name_file_start)

starting_dataset = starting_dataset.replace(np.NaN, "")


mmyssing = ["a", "x", "xc", "xr", "nc", "c", "s",0,1,2]

#Check the Flag value
check = starting_dataset[starting_dataset["Reference year"] == 2017].copy()
for i in tqdm(check.index):
    for var in diz_variabili_colonne_modifiche:

        flag = check.loc[i][diz_variabili_colonne_modifiche[var][3]]
        val_init = check.loc[i][diz_variabili_colonne_modifiche[var][4]]
        val_final = check.loc[i][diz_variabili_colonne_modifiche[var][5]]
        
        if  val_init == "m" and  flag == "" and val_final != "" and val_final not in mmyssing and val_final != "m":
            starting_dataset[diz_variabili_colonne_modifiche[var][3]][i] = "Smooth"
        elif val_init in mmyssing and  flag == "" and val_final != "" and val_final not in mmyssing and val_final != "m":
            starting_dataset[diz_variabili_colonne_modifiche[var][3]][i] = "Smooth Change"





starting_dataset.to_excel(name_file_final, sheet_name='Sheet_name_1',index = False)

imputation_test = pd.read_excel(name_file_final, sheet_name='Sheet_name_1')



#Function to Add Final Ratio to all the observations

def addRatio(dataset, var_1, var_2, name_col):
    lista_da_attaccare = []
    for i in tqdm(list(dataset.index)):
        num = imputation_test.loc[i][var_1]
        den = imputation_test.loc[i][var_2]
        
        try:
            rapporto = num/den
            lista_da_attaccare.append(rapporto)
        except:
            lista_da_attaccare.append("")
            
    dataset[name_col] = lista_da_attaccare
    
    return dataset




#Add Final Ratios into the Dataset

imputation_test = addRatio(imputation_test, 'Value Donor Graduates','Value Donor Enrolled', "Ratio Laureati/Iscritti")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (HC)','Value Donor Enrolled', "Ratio Academic Staff (HC)/Iscritti")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (FTE)','Value Donor Enrolled', "Ratio Academic Staff (FTE)/Iscritti")

imputation_test = addRatio(imputation_test, 'Value Donor Expenditure (EURO)','Value Donor Enrolled', "Ratio Spese/Iscritti")

imputation_test = addRatio(imputation_test, 'Value Donor Revenues (EURO)','Value Donor Enrolled', "Ratio Ricavi/Iscritti")

imputation_test = addRatio(imputation_test, 'Value Donor Expenditure (EURO)','Value Donor Revenues (EURO)', "Ratio Spese/Ricavi")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (FTE)','Value Donor Academic Staff (HC)', "Ratio Academic FTE/HC")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (HC)','Value Donor Graduates', "Ratio Academic Staff (HC)/Laureati")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (FTE)','Value Donor Graduates', "Ratio Academic Staff (FTE)/Laureati")

imputation_test = addRatio(imputation_test, 'Value Donor Expenditure (EURO)','Value Donor Graduates', "Ratio Spese/Laureati")

imputation_test = addRatio(imputation_test, 'Value Donor Revenues (EURO)','Value Donor Graduates', "Ratio Ricavi/Laureati")


imputation_test = addRatio(imputation_test, 'Value Donor Non Academic Staff (HC)','Value Donor Enrolled', "Ratio Non Academic Staff (HC)/Iscritti")

imputation_test = addRatio(imputation_test, 'Value Donor Non Academic Staff (FTE)','Value Donor Enrolled', "Ratio Non Academic Staff (FTE)/Iscritti")

imputation_test = addRatio(imputation_test, 'Value Donor Non Academic Staff (HC)','Value Donor Graduates', "Ratio Non Academic Staff (HC)/Laureati")

imputation_test = addRatio(imputation_test, 'Value Donor Non Academic Staff (FTE)','Value Donor Graduates', "Ratio Non Academic Staff (FTE)/Laureati")

imputation_test = addRatio(imputation_test, 'Value Donor Non Academic Staff (FTE)','Value Donor Non Academic Staff (HC)', "Ratio Non Academic FTE/HC")


#Ratios add only if the user is considering also "bibliometrics" variables.

imputation_test = addRatio(imputation_test, 'Value Donor PhD Graduates','Value Donor PhD Enrolled', "Ratio PhD Laureati/ PhD Iscritti")

if research_analysis == "1":

    imputation_test = addRatio(imputation_test, 'p','Value Donor Academic Staff (FTE)', "Ratio Publication/Academic Staff FTE")
    
    imputation_test = addRatio(imputation_test, 'p','Value Donor Academic Staff (HC)', "Ratio Publication/Academic Staff HC")
    
    imputation_test = addRatio(imputation_test, 'p','Value Donor PhD Enrolled', "Ratio Publication/ PhD Student")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (FTE)','Value Donor PhD Enrolled', "Academic Staff FTE/ PhD Student")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (HC)','Value Donor PhD Enrolled', "Academic Staff HC/ PhD Student")

imputation_test = addRatio(imputation_test, 'Value Donor PhD Enrolled','Value Donor Enrolled', "PhD Student/ Student")

imputation_test = addRatio(imputation_test, 'Value Donor PhD Graduates','Value Donor Graduates', "PhD Graduates/ Graduates")

imputation_test = addRatio(imputation_test, 'Value Donor PhD Enrolled','Value Donor Graduates', "PhD Student/ Graduates")




#Add Trend values in the final Dataset

#Compute the SLOPE of the LR Model
def calcoloCoeffAngolare_aggiuntaExcel(lista, anni):
    missing = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m"]
    
    lista_f = []
    anno_f = []

    for i in range(len(lista)):
        #print(i)
        if lista[i] not in missing:
            lista_f.append(lista[i])
            anno_f.append(anni[i])
    if len(lista_f) >= 2:        
        x = np.array(anno_f).reshape((-1, 1))
        y = np.array(lista_f).reshape(-1,1)
        try:
            #model = LinearRegression()
            #model.fit(x, y)
            model = LinearRegression().fit(x, y)
            return (model.coef_[0]/(sum(lista_f)/len(lista_f)))[0]    
        except:
            return ""
    else:
        return ""
    




working_variable = ['Value Donor Enrolled',
'Value Donor Graduates',"Value Donor PhD Enrolled",
                        "Value Donor PhD Graduates",
'Value Donor Academic Staff (FTE)',
'Value Donor Academic Staff (HC)',
'Value Donor Non Academic Staff (FTE)',
'Value Donor Non Academic Staff (HC)',
'Value Donor Expenditure (EURO)',
'Value Donor Revenues (EURO)']


diz_eterid_trend = {}

for etid in tqdm(set(imputation_test["ETER ID"])):
    diz_eterid_trend[etid] = {}
    #print("Lavorando con: " + str(etid))
    check = imputation_test[imputation_test["ETER ID"] == etid].copy()
    for vv in working_variable :
        years = check["Reference year"].values
        lista_valori = check[vv].values
        valore = calcoloCoeffAngolare_aggiuntaExcel(lista_valori, years)
        diz_eterid_trend[etid][vv] = valore



#Add Empty Trend Columns

imputation_test["Trend Students"] = ""
imputation_test["Trend Graduates"] = ""
imputation_test["Trend PhD Students"] = ""
imputation_test["Trend PhD Graduates"] = ""
imputation_test["Trend Academic Staff (FTE)"] = ""
imputation_test["Trend Academic Staff (HC)"] = ""
imputation_test["Trend Non Academic Staff (FTE)"] = ""
imputation_test["Trend Non Academic Staff (HC)"] = ""
imputation_test["Trend Expenditure (EURO)"] = ""
imputation_test["Trend Revenues (EURO)"] = ""


#Extract all the Trends variables
var_for_trend = [i for i in imputation_test.columns if "Trend" in i]
#var_for_trend



diz_compatibility_trend_value = {"Trend Students":'Value Donor Enrolled',
"Trend Graduates":'Value Donor Graduates',
"Trend PhD Students":"Value Donor PhD Enrolled" ,                           
"Trend PhD Graduates":"Value Donor PhD Graduates" ,                               
                                 
"Trend Academic Staff (FTE)":'Value Donor Academic Staff (FTE)',
"Trend Academic Staff (HC)":'Value Donor Academic Staff (HC)',
"Trend Non Academic Staff (FTE)":'Value Donor Non Academic Staff (FTE)',
"Trend Non Academic Staff (HC)":'Value Donor Non Academic Staff (HC)',
"Trend Expenditure (EURO)":'Value Donor Expenditure (EURO)',
"Trend Revenues (EURO)":'Value Donor Revenues (EURO)'}




#Function to Add TREND value
def addTrend(variabile_working, dizionario_valori_trend, data, diz_compatibility):
    for indice in tqdm(data.index):
        eter_id = data.loc[indice]["ETER ID"]
        
        for variable in variabile_working:
            val_to_add = diz_eterid_trend[eter_id][diz_compatibility[variable]]
            data[variable][indice] = val_to_add

    return data
        
        

imputation_test = addTrend(var_for_trend, diz_eterid_trend, imputation_test, diz_compatibility_trend_value)




imputation_test.to_excel(name_file_final, sheet_name='Sheet_name_1',index = False)


imputation_test = pd.read_excel(name_file_final, sheet_name='Sheet_name_1')


# =============================================================================
# Add Final Columns Name
# =============================================================================

  
if research_analysis == "1" : 
    with open("columns_ordered_bibliometric.pkl", "rb") as fp:   # Unpickling
        colonne_finali_ordered_by = pickle.load(fp)
        
elif research_analysis == "0":
    with open("columns_ordered.pkl", "rb") as fp:   # Unpickling
        colonne_finali_ordered_by = pickle.load(fp)
    
imputation_test = imputation_test[colonne_finali_ordered_by]



### Last Check for the uncorrect Flags

for i in tqdm(imputation_test.index):
    for vv in diz_variabili_colonne_modifiche:
        if imputation_test.loc[i][diz_variabili_colonne_modifiche[vv][3]] == "Donor Change" and imputation_test.loc[i][diz_variabili_colonne_modifiche[vv][5]] == "m":
                
                imputation_test[diz_variabili_colonne_modifiche[vv][3]][i] = ""
                
                imputation_test[diz_variabili_colonne_modifiche[vv][5]][i] = imputation_test.loc[i][vv]
                


### Remove Null values within the variables of the Flags
for i in imputation_test.columns:
    if "Flag" in i:
        imputation_test[i] = imputation_test[i].replace(np.nan,"")


### Correction Format

m_f = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m"]

for i in tqdm(imputation_test.index):
    for vv in diz_variabili_colonne_modifiche:
        if imputation_test.loc[i][diz_variabili_colonne_modifiche[vv][3]] == "":
            
            if imputation_test.loc[i][vv] not in m_f :
            
                if imputation_test.loc[i][diz_variabili_colonne_modifiche[vv][4]] not in m_f :

                    imputation_test[diz_variabili_colonne_modifiche[vv][5]][i] = imputation_test.loc[i][vv]

 

### Modify columns name according to the Excel file "Columns_Final_Name.xlsx"

final_col_name = []
for i in imputation_test.columns:
    if "Unnamed" not in i:
        final_col_name.append(i)

imputation_test = imputation_test[final_col_name]

col = pd.read_excel("Columns_Final_Name.xlsx")

if research_analysis == "1":
    col_final_names = col["Nomi Nuovi Finali"]
elif research_analysis == "0":
    col_names = col["Nomi Nuovi Finali No Research"]
    col_final_names = [x for x in col_names if str(x) != 'nan']

### Here we change the columns name
imputation_test.columns = col_final_names

### Last Check for this modification
imputation_test.head()

### Save new dataset
imputation_test.to_excel(name_file_final, sheet_name='Sheet_name_1',index = False)

print()
print("Total Imputation Completed")
print() 


