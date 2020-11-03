# -*- coding: utf-8 -*-
"""
@author: Davide Aureli, Renato Bruni, Cinzia Daraio 

"""

# =============================================================================
# LIBRARY
# =============================================================================

import pandas as pd
from pandas import DataFrame
import numpy as np
from collections import Counter
import math
import warnings
import os
import sys
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# PARAMETERS - Defined here, the user can change them
# =============================================================================

#HERE Research Analysis: (0 = No ; 1= Yes)
research_analysis = str(input("Does the dataset contain also the Research Information ? (Write 0 for 'NO' otherwise 1 for 'YES'):\n\n"))

#Save the answer of the User, which will be red also in all files
with open('./research_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(research_analysis)

#HERE the user specifies the name of the column under imputation
#possible choices: 0,1,2,3,4,5,6,7,8,9 

list_imputed_variables = [0,1,2,3,4,5,6,7,8,9]

for num_var_imput in list_imputed_variables:
    
    imp_variable = num_var_imput
    
    selected_variable = {
    0:'Total students enrolled ISCED 5-7',
    1:'Total graduates ISCED 5-7',
    2:'Total students enrolled at ISCED 8',
    3:'Total graduates at ISCED 8',
    4:'Total academic staff (FTE)',
    5:'Total academic staff (HC)', 
    6:'Number of non-academic  staff (FTE)',
    7:'Number of non-academic staff (HC)',
    8:'Total Current expenditure (EURO)', 
    9:'Total Current revenues (EURO)'}
    imputed_variable = selected_variable[imp_variable]
    
    
    #Simple Names
    simple_names = {
    0:'students',
    1:'graduates',
    2:'phd students',
    3:'phd graduates',
    4:'academic staff FTE',
    5:'academic staff HC', 
    6:'non academic staff FTE',
    7:'non academic staff HC',
    8:'expenditure', 
    9:'revenues'}
    
    #the type of the imputed variable, integer or decimal
    type_option = {
    0:'Int',
    1:'Int',
    2:'Int',
    3:'Int',
    4:'Dec',
    5:'Int',
    6:'Dec',
    7:'Int',
    8:'Int',
    9:'Int'}
    
    #Create the Directory where we save all the Imputation files. This is optimal for the First 
    #Smooth Imputation; while during the second part after donor Imputation we have not to create 
    #this folder. Define the name of the directory to be created.
    path = "./output_smooth/"
    
    #Save the path into a file. (We read it from another file for the final merging)
    with open('./path_smooth.txt', 'w', encoding='utf-8') as f:
        f.write(path)
    
    #directory and name of the saved file.
    name_file_saved = path + "fileout_"+ simple_names[imp_variable] +".xlsx"
    
    
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s already created" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
    
    #This is a Flag to define the insert of imputed values into an integer number ("Int") or decimal number("Dec").
    flag_int_dec = type_option[imp_variable]
    
    
    #HERE the user can choose the Exponent to smooth the negative values imputed during the Smooth Code,
    #The idea is to propagate in a smoothing way the starting value according to the SmoothExponent
    SmoothExponent = 0.6
    #default value 0.6
    

    
    # =============================================================================
    # READING DATASET
    # =============================================================================
    
    #Specify the input dataset
    starting_dataset = pd.read_excel(".\original_dataset.xlsx")
    
    
    #Rows and Columns
    starting_dataset.shape
    
    interesting_coloumns =starting_dataset.columns
    
    #First Check about the user's decision on the parameter "research_analysis"
    
    if research_analysis == "1":
        #Check if bibliometrics feature in columns:
        
        if "p" in interesting_coloumns and "pp(top 10)" in interesting_coloumns:
            #Ok 
            pass
        
        else:
            #This is the Warning Info for the User
            print()
            print("-- WARNING --")
            print("Bibliometric Information is NOT available in this File !s")
            print()
            
            sys.exit()
                   
        
        
    
    # Dataset selection
    
    #Select only the interesting columns.
    starting_dataset = starting_dataset[interesting_coloumns]
    #Number of rows
    print("Number of Observations: " + str(starting_dataset.shape[0]))
    
    
    #The most important feature is the Reference Year, and if an Institution has a None reference year we drop it.
    starting_dataset = starting_dataset.replace(np.NaN, "Null" )
    
    #Select only the rows with an accepted values for the Reference Year.
    starting_dataset = starting_dataset[starting_dataset["Reference year"] != "Null"]
    #print(starting_dataset.shape[0])
    
    
    #Missing Values all possibilities.
    missing_completi = ["m","a", "x", "xc", "xr", "nc", "c", "s", "Null"]
    
    print()
    print("Total Missing for: " + str(imputed_variable) + "\n")
    for i in missing_completi:
        print(i, Counter(starting_dataset[imputed_variable])[i])
        
    #Create a copy of the starting dataset at the end we will make the comparison, in this way we can modify only the copy.
    imputation_test = DataFrame(starting_dataset.copy())
    
    #We need of another copy because the observation will be deleted according to the different variables analyzed
    data_Excel = imputation_test.copy()
    
    # We transform different missing values as "m", our methodology will work on it.
    imputation_test[imputed_variable] = imputation_test[imputed_variable].replace(["x", "xc", "xr", "c", "s"], "m" )
    
    
    ### In this part we transform the isolated "a", sorrounded by "m" values into "m".
    
    for possibleID in set(imputation_test["ETER ID"]):
        
        #Extract a single institution
        check = imputation_test[imputation_test["ETER ID"] == possibleID].copy()
        
        #Here the variable is the one we are computing
        vv = imputed_variable
            
        valori_analisi = list(check[vv].values)
        indici_analisi = list(check[vv].index)
    
        numero_a = Counter(valori_analisi)["a"]
        numero_m = Counter(valori_analisi)["m"]
    
        if len (valori_analisi) > 1:
            if numero_m < len(valori_analisi) -1 and  numero_a == 1 and numero_m >= 1 and len(valori_analisi) - (numero_a + numero_m) >= 2 :
                
                #print(possible ETER ID)
                #print(vv)
                #print(valori_analisi)
                #print(indici_analisi)
                #print("INDEX to Change")
                #print(indici_analisi[valori_analisi.index("a")])
                
                ind = indici_analisi[valori_analisi.index("a")]
    
                #Excluded cases where "a" is in the first or last position
                #if ind == indici_analisi[-1] or ind == indici_analisi[0]:
                
                #print("Changed")
                imputation_test[vv][ind] = "m"
    
                #print()
                #print()
    
    
    #List of missing different from "m" and we need to remove from our analysis
                
    #Here we find the ETER ID with a null values not consider for the imputation and we delete it.
    
    replace = ["a", "x", "xc", "xr", "nc", "c", "s", "Null"]
    
    imputation_test = imputation_test.replace(replace, "eliminare" )
    print("Old Shape composed by: " + str(imputation_test.shape[0]) + " \n")
    
    print("Number of ETER ID to delete: ")
    print(len(imputation_test[imputation_test[imputed_variable] == "eliminare"]["ETER ID"]))
    print()
    eterID_delete = list(set(imputation_test[imputation_test[imputed_variable] == "eliminare"]["ETER ID"]))
    
    #Delete Observation with "eliminare"
    
    #imputation_test = imputation_test[imputation_test[imputed_variable] != "eliminare"]
    imputation_test = imputation_test[~imputation_test["ETER ID"].isin(eterID_delete)]
    print("New Shape composed by: " + str(imputation_test.shape[0]))
    
    print()
    print("Total Missing, after replacing values, for " + str(imputed_variable) + "\n")
    for i in missing_completi:
        #print(i, Counter(starting_dataset["Total students enrolled ISCED 5-7"])[i])
        print(i, Counter(imputation_test[imputed_variable])[i])
        
        
    #Extraction of the University dividing them between all missing values and only sequence.
    
    #All the institution with some missing values
    ist_selezionate = []
    #Total missing will be handled by Donor Code
    ist_selezionate_solo_missing = []
    
    #Use the ETER ID to select University cause some Universities change the Institution Name
    for ist in set(imputation_test["ETER ID"]):
        check = imputation_test[imputation_test["ETER ID"] == ist].copy()
        valori = check[imputed_variable].values
        #Some missing
        if Counter(valori)["m"] < len(valori) - 1 :
            ist_selezionate.append(ist)
        #All missing
        elif Counter(valori)["m"] == len(valori) or Counter(valori)["m"] == len(valori)-1 :
            ist_selezionate_solo_missing.append(ist)
            
    print()
    print("Working with: " + imputed_variable + "\n")
    print("Number of Institution (ETER ID) with some missing values: " + str(len(ist_selezionate)))
    print("Number of Institution (ETER ID) with all missing values: " + str(len(ist_selezionate_solo_missing)))
    print()
    
    #We extract only the Institution with some missing values (sequence of "m")
    method_first = imputation_test[imputation_test["ETER ID"].isin (ist_selezionate)].copy()
    
    #Dimension of the dataset, we do not consider the Institution with all missing.
    method_first.shape
    
    #Missing to be imputed.
    print("Number of Missing values 'm': " + str(Counter(method_first[imputed_variable])["m"]))
    
    
    # =============================================================================
    # IMPUTATION Code - Smooth Part
    # =============================================================================
    
    # Function to check the sequence of consecutive Missing.
    
    def checkSequenceMissing(dictionary):
        
        #Lista values 
        lista_val = list(dictionary.values())
        #Index first missing
        start_index = lista_val.index("m")
        #Number of Missing
        tot_missing = Counter(dictionary.values())["m"]
        #Boolean val for the missing sequence
        sequence = "OK"
    
        for i in range(tot_missing):
            if lista_val[start_index] != "m":
                #Here we detect not a consecutive missing values
                sequence = "Not consecutive"
            else:
                start_index += 1
        
        return sequence
    
    # SMOOTH Function
        
    #imputed value using linear regression
    from sklearn import  linear_model
    
    #We focus the attention on th sequence of "m" missing values.
    
    #The year are selected according to the data available
    def imputation_missing_LR(dataset, variable, year=[2017, 2016, 2015,2014,2013,2012,2011]):
        
        not_consecutive_missing = 0
        tutti_coef = []
        
        # Working considering the ETER ID
        for institute in tqdm(sorted(list(set(dataset["ETER ID"])))):
        #for institute in sorted(list(set(dataset["ETER ID"]))):
            
            #print("Working with this Institute: " + str(institute))
            #print()
            
            #Extract a small dataset about the info of that institute
            a = DataFrame(dataset[dataset["ETER ID"] == institute].copy())
            
            #Put the variable for imputation, osserviamo i valori di anno e variabile
            d = DataFrame(a[["Reference year",variable]].copy())
            
            #Dictionary Creation {Year:value: for instance 2016:31, 2015:20, 2014:m......2011:11}
            #valori = dict(sorted(d.values.tolist()))
            valori = dict(d.values.tolist())
    
            #Function transforms missing value "m", we need of at least 2 real values to make our prediction
            #Check about it and the presence of at least 1 element in the dictionary
                               
            if len(valori.keys()) >= 1 and len(valori.keys()) <=7 and Counter(list(valori.values()))["m"] >=1 and Counter(list(valori.values()))["m"] <= len(valori) - 2 :
            
                #print("We work with: " + str(institute))
                #print(valori)
    
                if any(i in valori.values() for i in replace) == False:
                    #print("This Institue "+ institute + "does not have missing different from m")
                    seq = checkSequenceMissing(valori)
                
                    #Here Institution with missing in differnet years not consecutive
                    if seq == "Not consecutive":
                        #Used the weighted average
                        fattore_peso = 2
                        not_consecutive_missing += 1
                        #print(institute)
                        #print(valori)
                    #Consecutive Missing
                    else:
                        #Used the weighted average
                        fattore_peso = 10
                        
                    
                    pp = a[["English Institution Name","Reference year",variable]]
                    pp = pp.sort_values(by=["Reference year"])
                    
                    # LINEAR REGRESSION
                    
                    Test = []
                    Train = []
                    
                    
                    indici = list(pp.index.values)
                    valori_cambiati_indici = []
    
                    for i in range(len(pp)):
                        #2 is the column imputed
                        if pp.iloc[i,2] == "m":
                            Test.append(i)
                            valori_cambiati_indici.append(indici[i])
                        else:
                            Train.append(i)
    
    
                    #Training --> Rows with data
                    #Test --> Rows with missing
                    
                    #1 is our X variable, in this case will be the Reference Year
    
                    X_train = pp.iloc[Train,1]
                    X_test = pp.iloc[Test,1]
    
                    y_train = pp.iloc[Train,2]
                    y_test = pp.iloc[Test,2]
    
    
    
                    #Make the Reshape, cause X must be a Matrix
                    
                    # Create linear regression object
                    regr = linear_model.LinearRegression()
    
                    # Train the model using the training set
                    regr.fit(np.array(X_train).reshape(-1,1), y_train)
    
                    # Make predictions using the testing set
                    pred = regr.predict(np.array(X_test).reshape(-1,1))
    
                    #From an array to list our prediction
                    pred = [int(elem) for elem in list(pred) ]
                    
                    # WEIGHTED AVERAGE 
    
                    if Test[0] > Train[-1]:
                        
    
    
                        numeratore = 0
                        peso = 1
                        tot_peso = 0
    
                        for elem in y_train:
                            
                            peso *= fattore_peso
                            numeratore += peso*elem
                            tot_peso += peso
    
                        media_pesata = numeratore/tot_peso
    
                    elif Test[0] < Train[-1]:
                        
    
                        numeratore = 0
                        peso = len(y_train)*1000
                        tot_peso = 0
    
                        for elem in y_train:
                            tot_peso += peso
                            numeratore += peso*elem
                            #peso = peso/10
                            peso = peso/fattore_peso
                            
                        media_pesata = int(numeratore/tot_peso)
                    #print()
                    #print("Starting Values")
                    #print(valori)
                    #print()
                    #print("Coeff (Slope) for LR Model")
                    #print(regr.coef_)
                    #print("Linear Regression Prediction")
                    #print(pred)
                    #print("Weighted Average Prediction")
                    #print(media_pesata)
                    #print()
    
                    #We need to normalize the angular coefficient for the growth, going from 1 to 1000 is different
                    #respect starting from 1000 to 2000
                    
                    normalizzatore = min(y_train)
    
                    #Here we make a check for the infinite values, in case where the min is 0
                    if normalizzatore == 0:
                        normalizzatore = 1
                    
                    
                    #Our Coefficient "a" (Coefficient "a" in Imputation between LR and WA)
                    
                    
                    #All Variables except PhD
                    coeff = 2* abs(regr.coef_) / normalizzatore
    
    
                    tutti_coef.append(coeff)
    
                    #print("Our coeff_a is:")
                    #print(coeff )
    
                    #Imputation between Linear Regression(LR) and Weighted Average(WA)
                    #Trend Smooth =  [a^2 / (a^2 + 1)](WA) +  [1 / (a^2 + 1)](LR)
    
                    final_val_imputed = []
    
                    for i in range(len(pred)):
                        vvv = pred[i]
                        if vvv < 0 :
                            final_val_imputed.append(vvv)
                        else:
                            linear_comb = (coeff[0]**2/(coeff[0]**2 +1 ))*media_pesata + (1/(coeff[0]**2 +1 ))*pred[i]
                            final_val_imputed.append(linear_comb)
    
                    
    #                print()
    #                print("Value Imputed")
    #                print(final_val_imputed)
    #                print()
                    
                    for i in range(len(valori_cambiati_indici)):
                        dataset[variable][valori_cambiati_indici[i]] = round(final_val_imputed[i],2)
                    
                            
        #print("Not consecutive Missing working on: " + imputed_variable)
        #print(not_consecutive_missing)
                    
        return dataset, tutti_coef
    
    #Smooth Imputation
    data_imputed, coef = imputation_missing_LR(method_first, imputed_variable)
    
    print()
    print("After Imputation number of remaining missing: ")
    print(Counter(data_imputed[imputed_variable])["m"])
    
    print()
    print("We work on this number of Institution: " + str(len(ist_selezionate)))
    
    #Compute % of ETER ID missing values
    print()
    print("We imput this percentage of missing val % :")
    print(((Counter(imputation_test[imputed_variable])["m"] - Counter(data_imputed[imputed_variable])["m"]) / Counter(imputation_test[imputed_variable])["m"])*100)
    
    
    ### Add the values in the imputed variable column to create the Excel file
    
    #Normalization with integer number
    def normal_round(n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)
    
    
    #Add values into the dataframe used to create Excel File    
    
    colonna_linear_combination = []
    
    for indice in data_Excel.index:
        try:
            if data_Excel.loc[indice][imputed_variable] != "a":
                
                if data_Excel.loc[indice][imputed_variable] in missing_completi:
                
                    if type_option[imp_variable] == "Int":
                    
                        colonna_linear_combination.append(normal_round(data_imputed.loc[indice][imputed_variable]))
                        
                    elif type_option[imp_variable] == "Dec":
                        
                        colonna_linear_combination.append(data_imputed.loc[indice][imputed_variable])
                #This other condition is related to the floating value
                else:
                    
                    colonna_linear_combination.append(starting_dataset.loc[indice][imputed_variable])
            else:
                #print(data_Excel.loc[indice]["ETER ID"])
                colonna_linear_combination.append("a")
        except:
            colonna_linear_combination.append(starting_dataset.loc[indice][imputed_variable])  
    
    
    
    #Add Final column to the Excel File.
    data_Excel["Imputation"] = colonna_linear_combination
    
    #Save the data into the Excel file. ("name_file_saved" specified at the beginning)
    
    data_Excel.to_excel(name_file_saved, sheet_name='Sheet_name_1', index = False)
    
        
    # =============================================================================
    # ### Read the Excel file to detect some negative values we have to "sweeten" 
    # =============================================================================
            
    #Reading the file created just before.
    starting_smooth = pd.read_excel(name_file_saved)   
    
    print("Total Missing, after first step of Smooth, for " + str(imputed_variable) + "\n")
    for i in missing_completi:
        print(i, Counter(starting_smooth["Imputation"])[i])
        
        
    # Now we transform again different missing values into "m"
    starting_smooth["Imputation"] = starting_smooth["Imputation"].replace(["x", "xc", "xr", "c", "s","nc"], "m" )
    
    
    print("Total Missing, after first step of Smooth, for " + str(imputed_variable) + "\n")
    for i in missing_completi:
        print(i, Counter(starting_smooth["Imputation"])[i])
        
        
        
    ### Here we see possible "a" values in the time series and we reconstruct "m" 
    ### in negative cases
           
    indici_cambio_finale = []
    
    for possibleID in set(starting_smooth["ETER ID"]):
        #print(possibleID)
        
        #Extract a single institution
        check = starting_smooth[starting_smooth["ETER ID"] == possibleID].copy()
        
        #Final variable to check
        vv = "Imputation"
            
        valori_analisi = list(check[vv].values)
        indici_analisi = list(check[vv].index)
        
        
        cc = 0
        for j in valori_analisi:
            if j not in missing_completi:
                if j<0:
                    cc += 1
                    
        
    
        numero_a = Counter(valori_analisi)["a"]
        numero_m = Counter(valori_analisi)["m"]
    
        if len (valori_analisi) > 1:
            if numero_a == 1 and cc > 0 :
    #            print(possibleID)
    #            print(vv)
    #            print(valori_analisi)
    #            print(indici_analisi)
    #            print("INDICE da Cambiare")
    #            print(indici_analisi[valori_analisi.index("a")])
                ind = indici_analisi[valori_analisi.index("a")]
    
                #We are excluding the cases where "a" is in the first or last position
                if ind != indici_analisi[-1] and ind != indici_analisi[0]:
                    #print("Changed")
                    starting_smooth[vv][ind] = "m"
                    
                    indici_cambio_finale.append(ind)
        
                    #print()
                    #print()   
    
                
    # =============================================================================
    # Sweeten Negative Prediction
    # =============================================================================
    
                 
    def changeNegativeValues(prova):
        
        indici_negativi = []  
       
        for ind, vv in enumerate(prova):
            if vv not in missing_completi and vv <0:
                indici_negativi.append(ind)
        
        #print(indici_negativi)
        
        if len(prova)-1 in indici_negativi or len(prova)-2 in indici_negativi:
            
            
            for i in range(len(indici_negativi)):
                prova[indici_negativi[i]] = int(prova[indici_negativi[i]-1]*(SmoothExponent))
                
        else:
        
            for i in range(len(indici_negativi)-1,-1,-1):
                
                prova[indici_negativi[i]] = int(prova[indici_negativi[i]+1]*(SmoothExponent))
                
                
        return prova
    
    for universita in sorted(list(set(starting_smooth["ETER ID"]))):
        
        #Values and Index
        valori_in = starting_smooth[starting_smooth["ETER ID"] == universita]["Imputation"].values
        indici = list(starting_smooth[starting_smooth["ETER ID"] == universita]["Imputation"].index)
        
        annual = starting_smooth[starting_smooth["ETER ID"] == universita]["Reference year"].values
        
        negativo = False
        
        for elem in valori_in:
            if elem not in missing_completi:
                if elem <0 :
                    #print(universita)
                    #print(elem)
                    
                    negativo = True
                    
        if negativo:
            
            #print("Negative Case")
            #print(universita) 
            #print(valori_in)
            #print(indici)
            
            
            #print("Starting Imputation")
            final_value_negative = changeNegativeValues(valori_in)
            #print(final_value_negative)
            
            #We change the values also in valori_in so we need to re-extract them
            valori_in = starting_smooth[starting_smooth["ETER ID"] == universita]["Imputation"].values
        
            
            for ind_j in range(len(indici)):
                
                #Check about the change in the values
                if valori_in[ind_j] != final_value_negative[ind_j]:
                    
                    #print(valori_in[ind_j])
                    #print(final_value_negative[ind_j])
                    
                    #Update Negative values into the smooth one
                    starting_smooth["Imputation"][indici[ind_j]] = final_value_negative[ind_j]
            
            #print("Finish")
            #print()
            #print()
            
    
    
    #We put "a" in the old values in the variable Imputation
    for ind in indici_cambio_finale:
        starting_smooth["Imputation"][ind] = "a"
    
    #Save just the ETER ID and Imputation Column in the Excel file. 
    starting_smooth = starting_smooth[["ETER ID", "Imputation"]]
        
    # Save The Results
    starting_smooth.to_excel(name_file_saved, sheet_name='Sheet_name_1',index = False)
    
print("Smooth Imputation Completed")
print()    
