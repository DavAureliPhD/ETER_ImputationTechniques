
# -*- coding: utf-8 -*-
"""
@author: Davide Aureli, Renato Bruni, Cinzia Daraio

"""

# =============================================================================
# LIBRARY
# =============================================================================
import os
import pandas as pd
import numpy as np
from collections import Counter
import math
import pickle
import warnings
warnings.filterwarnings("ignore")

#Distance Measurement
from sklearn.neighbors import NearestNeighbors

#Progress Bar
from tqdm import tqdm

#Seed value to obtain always the same donor imputation result.
import random
random.seed(42)


# =============================================================================
# PARAMETERS - Defined here, the user can change them
# =============================================================================

#Read Research_analysis parameter
with open('research_analysis.txt', encoding='utf-8') as f:
    research_analysis = f.read().replace('\n', '')
    
#Create the Directory where we save all the Imputation files. 
path = "./output_donor/"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

#Save the path into a file. (We read it from another file for the final merging)
with open('./path_imputation_donor.txt', 'w', encoding='utf-8') as f:
    f.write(path)

#directory and name of the saved file.
name_file_saved = path + "fileout_donor.xlsx"

# =============================================================================
# Reading Dataset
# =============================================================================

with open('path_smooth.txt', encoding='utf-8') as f:
    path = f.read().replace('\n', '')

starting_dataset = pd.read_excel(path + "./fileout_smooth_complete.xlsx")

#The most important feature is the Reference Year, and if an Institution has a None 
#reference year we drop it.
starting_dataset = starting_dataset.replace(np.NaN, "Null" )
starting_dataset = starting_dataset[starting_dataset["Reference year"] != "Null"]


# Modify strange cases where FTE values is 0 and HC is not 0. So we will update both to 0. 

#Definition of possible Missing Values
possible_missing = ["m","a", "x", "xc", "xr", "nc", "c", "s"]

#List of indexes to modify
ind = []
naz = []
conteggio = 0

for i in tqdm(starting_dataset.index):
    val_FTE = starting_dataset.loc[i]["Smooth Academic Staff FTE"]
    val_HC = starting_dataset.loc[i]["Smooth Academic Staff HC"]
    if val_FTE not in possible_missing and val_HC not in possible_missing:
        
        #Detect possible values not real according to the definition of FTE and HC
        if val_FTE == 0 and val_HC > 0:
            ind.append(i)
            naz.append(starting_dataset.loc[i]["Country Code"])
            conteggio += 1
            starting_dataset["Smooth Academic Staff HC"][i] = 0            


# =============================================================================
# SIMILARITY NATIONS - Add Feature 
# =============================================================================

#This part is useful for the computation on the distance to choose a possible Donor
#considering also the feature about Geographical Region.
            
similarity_country = {}

#All possible Country Code
country = set(list(starting_dataset["Country Code"]))

country = ["NL", "BE", "LU", "CH","LI", "DE", "AT", "IT", "ES", "GR", "PT", "HU", "CZ", "LT", "LV", "PL", "EE","SK", 
           "AL","BG","HR","ME","SI","MK","RS","RO", "NO", "SE", "DK", "FI","IS", "UK", "IE", "MT", "FR", "CY","TR"]
           

for naz in country:
    if naz in ["NL", "BE", "LU", "CH","LI"] :
        similarity_country[naz] = 1
    elif naz in ["DE", "AT"]:
        similarity_country[naz] = 2
    elif naz in ["IT", "ES", "GR", "PT"]:
        similarity_country[naz] = 3
    #EST del Nord
    elif naz in ["HU", "CZ", "LT", "LV", "PL", "EE","SK"]:
        similarity_country[naz] = 4
    #EST del SUD
    elif naz in ["AL","BG","HR","ME","SI","MK","RS","RO"]:
        similarity_country[naz] = 5
    elif naz in ["NO", "SE", "DK", "FI","IS"]:
        similarity_country[naz] = 6
    elif naz in ["UK", "IE", "MT"]:
        similarity_country[naz] = 7
    elif naz in ["FR"]:
        similarity_country[naz] = 8
    elif naz in ["CY","TR"]:
        similarity_country[naz] = 9
    else:
        #Check for the Nation without a specific region.
        print(naz)

#Save Similarity_Country Dictionary
with open('similarity_country.pkl', 'wb') as handle:
    pickle.dump(similarity_country, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
diz_nations_and_code = {}
stringa = "AT Austria AL Albania LI Liechtenstein BE Belgium LT Lithuania BG Bulgaria LU Luxembourg CH Switzerland LV Latvia CY Cyprus ME Montenegro CZ CzechRepublic MK FormerRepublicofMacedonia DE Germany MT Malta DK Denmark NL Netherlands EE Estonia NO Norway GR Greece PL Poland ES Spain PT Portugal FI Finland RO Romania FR France RS Serbia HR Croatia SE Sweden HU Hungary SI Slovenia IE Ireland SK Slovakia IS Iceland TR Turkey IT Italy UK UnitedKingdom"
stringa = stringa.split()
i = 0
while i <= len(stringa) -1:
    diz_nations_and_code[stringa[i]] =stringa[i+1] 
    i += 2
#print(diz_nations_and_code)


zoneGeographic_CountryCode = {}
for elem in set(similarity_country.values()):
    for k in similarity_country.keys():
        if similarity_country[k] == elem and elem not in zoneGeographic_CountryCode:
            zoneGeographic_CountryCode[elem] = [k]
        elif similarity_country[k] == elem and elem in zoneGeographic_CountryCode:
            zoneGeographic_CountryCode[elem].append(k)

print("The Geographical Area with the Nations that belong to this particular zone")
print()
            
for val in zoneGeographic_CountryCode:
    print("For the Geographic Area number: " + str(val))
    print()
    for elem in zoneGeographic_CountryCode[val]:
        print(diz_nations_and_code[elem])
    print()



#Function to add Similarity Nations

def add_similarity(dataframe):
    lista_sim_nations = []
    for i in tqdm(dataframe.index):
        #Extract the corresponding values of the similarity of country dictionary
        val = similarity_country[dataframe.loc[i]["Country Code"]]
        lista_sim_nations.append(val)
    dataframe.insert(loc=8, column='Similarity Nations', value=lista_sim_nations)
    return dataframe


# Create a copy of the starting dataset 
imputation_test = starting_dataset.copy()


# Add Similarity Nation 
imputation_test = add_similarity(imputation_test)

#Observing the first 5 rows of the Dataframe
imputation_test.head()



## We set an ORDER for Nations applied during the Donor Imputation.

#Ordering all Nations.
lista_naz_imp = ["NL", "BE", "LU", "CH","LI","DE", "AT","IT", "ES", "GR", "PT","NO", "SE",
                 "DK", "FI","IS","UK", "IE", "MT","FR"]
balcani_Nord = ["HU", "CZ", "LT", "LV", "PL", "EE","SK"]
balcani_Sud = ["AL","BG","HR","ME","SI","MK","RS","RO"]

#Sorting in an alphabetical order.
lista_naz_imp = sorted(lista_naz_imp)
balcani_Nord = sorted(balcani_Nord)
balcani_Sud = sorted(balcani_Sud)

#Union between Nations.
priority_for_donor_imputation = lista_naz_imp + balcani_Nord + balcani_Sud

#Adding Cyprus.
priority_for_donor_imputation.append("CY")
#Adding Turkey. 
priority_for_donor_imputation.append("TR")

#Print the last list with the complete order of the nations
#print(priority_for_donor_imputation)

with open('priority_for_donor_imputation.pkl', 'wb') as handle:
    pickle.dump(priority_for_donor_imputation, handle, protocol=pickle.HIGHEST_PROTOCOL)


# =============================================================================
# RATIOS - Correlated Variables
# =============================================================================

#The ratio is the relationship between 2 variables during a specific Reference years.
#(i.e. Graduates/Students or PhD Graduates/PhD Students)

## Function to add Ratio
def addRatio(dataset, var_1, var_2, name_col):
    lista_da_attaccare = []
    for i in tqdm(list(dataset.index)):
        num = imputation_test.loc[i][var_1]
        den = imputation_test.loc[i][var_2]

        try:
            rapporto = num/den
            
            if np.isnan(rapporto) or np.isinf(rapporto):
                #"Nada" detects an infinite value or the impossibility to compute the Ratio
                rapporto = "Nada"
            
            lista_da_attaccare.append(rapporto)
        except:
            lista_da_attaccare.append("Nada")
            
    dataset[name_col] = lista_da_attaccare
    
    return dataset




### Adding Ratio specifying Numerator and Denominator 


imputation_test = addRatio(imputation_test, 'Smooth Students Graduates','Smooth Students Enrolled', "Ratio Laureati/Iscritti")

imputation_test = addRatio(imputation_test, 'Smooth Academic Staff HC','Smooth Students Enrolled', "Ratio Academic Staff (HC)/Iscritti")

imputation_test = addRatio(imputation_test, 'Smooth Academic Staff FTE','Smooth Students Enrolled', "Ratio Academic Staff (FTE)/Iscritti")

imputation_test = addRatio(imputation_test, 'Smooth Expenditure (EURO)','Smooth Students Enrolled', "Ratio Spese/Iscritti")

imputation_test = addRatio(imputation_test, 'Smooth Revenues (EURO)','Smooth Students Enrolled', "Ratio Ricavi/Iscritti")

imputation_test = addRatio(imputation_test, 'Smooth Expenditure (EURO)', 'Smooth Revenues (EURO)', "Ratio Spese/Ricavi")

imputation_test = addRatio(imputation_test, 'Smooth Academic Staff FTE','Smooth Academic Staff HC', "Ratio Academic FTE/HC")

imputation_test = addRatio(imputation_test, 'Smooth Academic Staff HC','Smooth Students Graduates', "Ratio Academic Staff (HC)/Laureati")

imputation_test = addRatio(imputation_test, 'Smooth Academic Staff FTE','Smooth Students Graduates', "Ratio Academic Staff (FTE)/Laureati")

imputation_test = addRatio(imputation_test, 'Smooth Expenditure (EURO)','Smooth Students Graduates', "Ratio Spese/Laureati")

imputation_test = addRatio(imputation_test, 'Smooth Revenues (EURO)','Smooth Students Graduates', "Ratio Ricavi/Laureati")


imputation_test = addRatio(imputation_test, 'Smooth Non Academic Staff HC','Smooth Students Enrolled', "Ratio Non Academic Staff (HC)/Iscritti")

imputation_test = addRatio(imputation_test, 'Smooth Non Academic Staff FTE','Smooth Students Enrolled', "Ratio Non Academic Staff (FTE)/Iscritti")

imputation_test = addRatio(imputation_test, 'Smooth Non Academic Staff HC','Smooth Students Graduates', "Ratio Non Academic Staff (HC)/Laureati")

imputation_test = addRatio(imputation_test, 'Smooth Non Academic Staff FTE','Smooth Students Graduates', "Ratio Non Academic Staff (FTE)/Laureati")

imputation_test = addRatio(imputation_test, 'Smooth Non Academic Staff FTE','Smooth Non Academic Staff HC', "Ratio Non Academic FTE/HC")

imputation_test = addRatio(imputation_test, 'Smooth PhD Graduates','Smooth PhD Enrolled', "Ratio PhD Laureati/ PhD Iscritti")

# 3 Ratios below just if the user works considering also Bibliometrics data
if research_analysis == "1":
    
    imputation_test = addRatio(imputation_test, 'p','Smooth Academic Staff FTE', "Ratio Publication/Academic Staff FTE")

    imputation_test = addRatio(imputation_test, 'p','Smooth Academic Staff HC', "Ratio Publication/Academic Staff HC")

    imputation_test = addRatio(imputation_test, 'p','Smooth PhD Enrolled', "Ratio Publication/ PhD Student")

imputation_test = addRatio(imputation_test, 'Smooth Academic Staff FTE','Smooth PhD Enrolled', "Ratio Academic Staff FTE/ PhD Student")

imputation_test = addRatio(imputation_test, 'Smooth Academic Staff HC','Smooth PhD Enrolled', "Ratio Academic Staff HC/ PhD Student")

imputation_test = addRatio(imputation_test, 'Smooth PhD Enrolled','Smooth Students Enrolled', "Ratio PhD Student/ Student")

imputation_test = addRatio(imputation_test, 'Smooth PhD Graduates','Smooth Students Graduates', "Ratio PhD Graduates/ Graduates")

imputation_test = addRatio(imputation_test, 'Smooth PhD Enrolled','Smooth Students Graduates', "Ratio PhD Student/ Graduates")



### Sum Up values for all Institutions without ratios

# diz_reference_ratio = {ICS_1:{Ratio_1 : valore_1, Ratio_2: valore_2 ... }, ICS_2:{Ratio_1 : valore_1, Ratio_2: valore_2 ... }, ... }


diz_reference_ratio = {0:{}, 1:{}, 2:{}}

#All columns with "Ratio" in the name
colonne_rapporto = [ i for i in imputation_test.columns if "Ratio" in i]



#Observing average values of Ratio when the University has not that value

def extractMean(lista):
    lista_finale = [elem for elem in lista if elem != "Nada" and elem != 0]
    #Return Mean value
    try:
        valore_finale = sum(lista_finale)/len(lista_finale)
    
    except:
        valore_finale = 0
    return valore_finale




#Code for all ICS (Institution Category Standardized) - AVERAGE Value

#Cycle for all possible ICS
for i in [0,1,2]:
    #Extract data with a specific ICS
    data_work = imputation_test[imputation_test["Institution Category standardized"] == i]
    #For with all the Ratio columns
    for name_col in colonne_rapporto:
        valori = list(data_work[name_col].values)
        diz_reference_ratio[i][name_col] =extractMean(valori)

#Print the dictionary with average values
#print(diz_reference_ratio)



#Observing variance values of Ratio when the University has not that value

def extractVariance(lista):
    lista_finale = [elem for elem in lista if elem != "Nada" and elem!= 0]
    valore_finale = np.var(lista_finale)
    
    return valore_finale


#Code for all ICS - VARIANCE Value

#Same work as before - Now we compute the variance

diz_reference_ratio_variance = {0:{}, 1:{}, 2:{}}

for i in [0,1,2]:
    #print(i)
    data_work = imputation_test[imputation_test["Institution Category standardized"] == i]
    
    for name_col in colonne_rapporto:
        valori = list(data_work[name_col].values)
        diz_reference_ratio_variance[i][name_col] =extractVariance(valori)

#Print the dictionary with variance values
#print(diz_reference_ratio_variance)




# Change sequence with 0,1 or 2 singular values surrounded by "m" into complete missing sequence.
# Remember after the imputation part, we will retrieve the beginning values and change them.
        
missing_check_restructure = ["a", "x", "xc", "xr", "nc", "c", "s"]

#Values with real value as 0,1, or 2
variabili_di_lavoro = ['Total students enrolled ISCED 5-7','Total graduates ISCED 5-7','Total students enrolled at ISCED 8',
                       'Total graduates at ISCED 8',                       
                       'Total academic staff (FTE)','Total academic staff (HC)', 
       'Number of non-academic  staff (FTE)','Number of non-academic staff (HC)',
        'Total Current expenditure (EURO)',
       'Total Current revenues (EURO)'
       ]


#After the first imputation we transform previous values into "m"

#Values to change 
variabili_connesse = ['Smooth Students Enrolled',
       'Smooth Students Graduates', 'Smooth PhD Enrolled', 'Smooth PhD Graduates', 'Smooth Academic Staff FTE',
       'Smooth Academic Staff HC', 'Smooth Non Academic Staff FTE',
       'Smooth Non Academic Staff HC', 'Smooth Expenditure (EURO)',
       'Smooth Revenues (EURO)']


#The idea is to set "m" into the beginning values, observing our imputation from Smooth Results,
#becuase in the column of Smooth we have "m" for each possible misisng values 
#we wanted to reconstruct

#Code to check this situation

for i in tqdm(imputation_test.index):
    
    for j in range(len(variabili_di_lavoro)):
    
        if imputation_test.loc[i][variabili_di_lavoro[j]] in missing_check_restructure and imputation_test.loc[i][variabili_connesse[j]]=="m":
            
            #print(imputation_test.loc[i][variabili_di_lavoro[j]])
            #print(imputation_test.loc[i][variabili_connesse[j]])
            #print()
            imputation_test[variabili_di_lavoro[j]][i] = "m"



#At the end these values should be reported into their starting value

variabili_di_lavoro = ['Total students enrolled ISCED 5-7','Total graduates ISCED 5-7','Total students enrolled at ISCED 8',
                       'Total graduates at ISCED 8',                       
                       'Total academic staff (FTE)','Total academic staff (HC)', 
       'Number of non-academic  staff (FTE)','Number of non-academic staff (HC)',
        'Total Current expenditure (EURO)',
       'Total Current revenues (EURO)', 'Smooth Students Enrolled',
       'Smooth Students Graduates', 'Smooth PhD Enrolled', 'Smooth PhD Graduates', 'Smooth Academic Staff FTE',
       'Smooth Academic Staff HC', 'Smooth Non Academic Staff FTE',
       'Smooth Non Academic Staff HC', 'Smooth Expenditure (EURO)',
       'Smooth Revenues (EURO)'
       ]

#Possible situation that we consider to modify into a complete sequence of missing

for possibleID in set(imputation_test["ETER ID"]):
    check = imputation_test[imputation_test["ETER ID"] == possibleID].copy()
    for vv in variabili_di_lavoro:
        
        valori_analisi = list(check[vv].values)
        indici_analisi = list(check[vv].index)
        
        #Here we take the count fro 0,1 or 2 which are sourrended by sequence of "m"
        numero_zeri = Counter(valori_analisi)[0]
        numero_uno = Counter(valori_analisi)[1]
        numero_due = Counter(valori_analisi)[2]
        numero_m = Counter(valori_analisi)["m"]
        
        if len (valori_analisi) > 1:
            if numero_m == len(valori_analisi) -1 and  numero_zeri == 1:
#                print(possibleID)
#                print(vv)
#                print(valori_analisi)
#                print(indici_analisi)
#                print("Index to Change")
#                print(indici_analisi[valori_analisi.index(0)])
                ind = indici_analisi[valori_analisi.index(0)]
                imputation_test[vv][ind] = "m"

                #print()
            elif numero_m == len(valori_analisi) -1 and numero_uno == 1 :
#                print(possibleID)
#                print(vv)
#                print(valori_analisi)
#                print(indici_analisi)
#                print("Index to Change")
#                print(indici_analisi[valori_analisi.index(1)])
                ind = indici_analisi[valori_analisi.index(1)]
                imputation_test[vv][ind] = "m"
#                print()
                
            elif numero_m == len(valori_analisi) -1 and numero_due == 1:
#                print(possibleID)
#                print(vv)
#                print(valori_analisi)
#                print(indici_analisi)
#                print("Index to Change")
#                print(indici_analisi[valori_analisi.index(2)])
                ind = indici_analisi[valori_analisi.index(2)]
                imputation_test[vv][ind] = "m"
#                print()


## The same will be applied to the value "a" in a complete sequence of "m".

for possibleID in set(imputation_test["ETER ID"]):
    check = imputation_test[imputation_test["ETER ID"] == possibleID].copy()
    for vv in variabili_di_lavoro:
        
        valori_analisi = list(check[vv].values)
        indici_analisi = list(check[vv].index)
        
        numero_a = Counter(valori_analisi)["a"]
        numero_m = Counter(valori_analisi)["m"]
        
        if len (valori_analisi) > 1:
            if numero_m == len(valori_analisi) -1 and  numero_a == 1:
#                print(possibleID)
#                print(vv)
#                print(valori_analisi)
#                print(indici_analisi)
#                print("Index to Change")
#                print(indici_analisi[valori_analisi.index("a")])
                ind = indici_analisi[valori_analisi.index("a")]
                
                #Excluding the cases where "a" is the first or the last element 
                if ind != indici_analisi[-1] and ind != indici_analisi[0]:
                    imputation_test[vv][ind] = "m"

                    #print()
                #print()




# Here we modify values for the variables Students **Enrolled** and **Graduates**, 
#that shows strange values according to a specific year.

#For each Institution
for i in set(imputation_test["ETER ID"]):
    
    #Possible missing
    miss = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m", 'eliminare', "Nada"]
    check = imputation_test[imputation_test["ETER ID"] == i]
    indici = check.index
    
    #Smooth Values
    vv1 = check["Smooth Students Enrolled"].values
    vv2 = check["Smooth Students Graduates"].values
    #Starting values
    r1 = check["Total students enrolled ISCED 5-7"].values
    r2 = check["Total graduates ISCED 5-7"].values
    
    for j in range(len(vv1)):
        if vv1[j] not in miss and vv2[j] not in miss:
            if vv1[j] == 0 and vv2[j] > 0:
                
#                print(i)
#                print(vv1[j],vv2[j])
#                print(r1[j],r2[j])
#                print(indici[j])
#                print()
        
                #We modify only the values imputed through the Smooth
                if r1[j] in miss and r2[j] in miss:
                
                    imputation_test["Smooth Students Graduates"][indici[j]] = 0  






### Save the dataset modified, after this preparation part we start the donor imputation.
imputation_test.to_excel(name_file_saved,index = False)

### Check made after the Smooth Part
check_dataset = pd.read_excel(name_file_saved)






def calcoloMedia (lista):
    #print("Starting List: ")
    #print(lista)
    
    miss = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m", 'eliminare', "Nada"]  
    
    tot = 0
    elementi = 0
    for e in lista:
        if e not in miss:
            tot += e
            elementi += 1
    if tot != 0:
        return tot/elementi
    else:
        return "Nada"

# Function to find out the indexes which do not respect the limit range 
# that we expect of the 40% window. In this case, the values that were given at 
# the beginning of the ETER data analysis will be used, then a second function
# will clean up the indices.
# 
# *var_1* = Numerator
# 
# *var_2* = Denominator
        
edge_window = 0.4

def estrazioneIndici_NonCorretti(dataset, var_1, var_2, nome_rapporto):
    
    indici_da_cambiare = []

    diz_indice_confine = {}

    for eterID in set(dataset["ETER ID"]):
        #print("Working with:")
        #print(eterID)

        #Select dataset to work
        prova = dataset[dataset["ETER ID"] == eterID].copy()
        indici = prova.index
        #print(prova["Ratio Laureati/Iscritti"].values)
        limite = calcoloMedia(prova[nome_rapporto].values)

        #Compute the mean of the variable on which we work.
        #If we find out a "Nada" as value we need to see the general values for ICS.

        if limite != "Nada":
            #print("The local limit value is: " + str(limite))
            for i in indici:

                #Here we compare our imputation respect to the value "limite"(boundaries) we 
                #have computed before.
                try:
                    valore_confronto = prova.loc[i][var_1]/prova.loc[i][var_2]
                    #print(valore_confronto)
                    if valore_confronto <= (limite - edge_window*limite) or valore_confronto >= (limite + edge_window*limite):

                        #print(i)
                        indici_da_cambiare.append(i)

                        diz_indice_confine[i] = limite
                except:
                    pass

        else:
            #Extract ICS of the candidate
            ics = prova.loc[indici[0]]["Institution Category standardized"]

            #Last check to be sure that the value is not missing
            if ics not in ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m", 'eliminare', "Nada"]:

                limite = diz_reference_ratio[ics][nome_rapporto]
                #print("The global limit value is: " + str(limite))
                for i in indici:

                    #Select our Ratio
                    try:
                        valore_confronto = prova.loc[i][var_1]/prova.loc[i][var_2]
                        #print(valore_confronto)
                        if valore_confronto <= (limite - edge_window*limite) or valore_confronto >= (limite + edge_window*limite):

                            #print(i)
                            indici_da_cambiare.append(i)

                            diz_indice_confine[i] = limite
                    except:
                        pass


        #print()
        
        
    return indici_da_cambiare, diz_indice_confine





#In this part we work only with Smooth of Graduates and Enrolled
indici_da_cambiare, diz_indice_confine = estrazioneIndici_NonCorretti(check_dataset, "Smooth Students Graduates", 
                                                                      "Smooth Students Enrolled", "Ratio Laureati/Iscritti")

print("Number of indeces to change: " + str(len(indici_da_cambiare)))


# Now we have to clean up the indexes that we extracted because some of these indexes 
#are linked to the data that ETER makes available


#Possible Missing Value
missing_typo = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m", 'eliminare', "Nada"]




#Function to clean a list of indeces
def puliziaIndici (dataset, lista_indici, var_1_iniziale, var_2_iniziale, lista_mancanti):
    #We have to verify that the indices we want to change are actually imputed by us
    final_index = []

    for indice in lista_indici:
        
        #var_1_iniziale = Numerator
        numeratore = dataset.loc[indice][var_1_iniziale]
        #var_2_iniziale = Denominator
        denominatore = dataset.loc[indice][var_2_iniziale]
        
        #Let's consider only those that have missing value on both of the starting variables
        #In this way we are sure to work with the values that we have imputed with smooth
        if numeratore in lista_mancanti and denominatore in lista_mancanti:
            final_index.append(indice)
            
    return final_index


#Final Index to modify in the dataset
final_index = puliziaIndici(check_dataset, indici_da_cambiare, "Total graduates ISCED 5-7", "Total students enrolled ISCED 5-7", missing_typo)

print("Number of indeces to modify after the analysis " + str(len(final_index)))




#Now we need a function able to detect which part of the windiw is exceeded by the value, if the
#UB (Upper Bound) or the LB (Lower Bound). Then we change the value on the numerator, for
#instance Graduates/Students will have some modification on the Graduates to bring back
#the ratio within the boundaries.

def cambioValore(dataset, indici_valori, dizionario_indice_valore, var_numeratore, var_denominatore):

    for elem in indici_valori:
        #Local value or Global value
        m = dizionario_indice_valore[elem]
        
        #Window = [l=lower; u=upper] - edge_window is a default parameter
        u = m + m*edge_window
        l = m - m*edge_window

        numeratore = dataset.loc[elem][var_numeratore]
        denominatore = dataset.loc[elem][var_denominatore]

        #print(elem)
        #print("Numerator: " + str(numeratore))
        #print("Denominator: " + str(denominatore))
        check = numeratore/denominatore
        
        #print("m: " + str(m))
        
        #Check about which part of the window is exceeded
        
        if check > u:
            #print("Upper")
            #print("New numerator")
            #print(u*denominatore)
            valore_aggiunto = u*denominatore
        elif check < l:
            #print("Lower")
            #print("New Numerator")
            #print(u*denominatore)
            valore_aggiunto = l*denominatore
            
        
        #Using this print to observe the values to change
        #print("Institution: " + str(dataset.loc[elem]["ETER ID"]))
        #print("New Value = " + str(valore_aggiunto))
        #print("Old Value = " + str(numeratore))
        dataset[var_numeratore][elem] = round(valore_aggiunto,0)
        
        #print()
        
    return dataset        

#We transform the values which go over the bounds into our limits according to the dictionary.
check_dataset = cambioValore(check_dataset, final_index, diz_indice_confine, "Smooth Students Graduates", "Smooth Students Enrolled")



### Save the new Dataset
check_dataset.to_excel(name_file_saved,index = False)



# =============================================================================
### DONOR IMPUTATION - Code
# =============================================================================

#Reading the dataset
imputation_test = pd.read_excel(name_file_saved)




#Dictionary that will be used at the end of the imputation part to normalize values imputed.


#________________________________________________________________________________

#Define variables used for the normalization

var_normalization = ['Smooth Students Enrolled',
       'Smooth Students Graduates', 'Smooth PhD Enrolled', 'Smooth PhD Graduates', 'Smooth Academic Staff FTE',
       'Smooth Academic Staff HC', 'Smooth Non Academic Staff FTE',
       'Smooth Non Academic Staff HC', 'Smooth Expenditure (EURO)',
       'Smooth Revenues (EURO)']

#______________________________________________________________________________

#Missing values in the normalization
miss_normalization = ["a", "x", "xc", "xr", "nc", "c", "s","m"]

#Legenda: 

#0 - Total Missing Variable
#1 - Only 1 Numeric Value
#2+ - Good Variable

diz_eterid_per_normalizzare = {nome:{"0":[], "1":[], "2o+":[]} for nome in set(imputation_test["ETER ID"])}

#Creation of a comulative dictionary summing up the situation for each ETER ID,
#to take into account which variables could be used to normalize 

for name in tqdm(diz_eterid_per_normalizzare.keys()):
    #print(name)
    #print(diz_eterid_per_normalizzare[name])
    check = imputation_test[imputation_test["ETER ID"] == name].copy()
    for variable in var_normalization:
        valori = check[variable].values
        #print(variable)
        #print(valori)
        totale = 0
        for j in miss_normalization:
            #print(Counter(valori)[j])
            totale += Counter(valori)[j]
        #print(totale)
        
        #Total Missing
        if totale == len(valori):
            diz_eterid_per_normalizzare[name]["0"].append(variable)
        #Only 1 numeric value
        elif totale == len(valori) -1:
            diz_eterid_per_normalizzare[name]["1"].append(variable)
        #Good variable
        else:
            diz_eterid_per_normalizzare[name]["2o+"].append(variable)

#Example:

#print("Small Example of one University: \n" )
#ETER ID = IT0005
#print(diz_eterid_per_normalizzare['IT0005'])



# =============================================================================
# Define ALL the Variables in the DONOR Imputation
# =============================================================================

# Now we define the variables on which we will work, 
# the correlation variables and finally we define the possible Donor Institutions.

#________________________________________________________________________________

#Define variable for the imputation


variabili_imputazione = ['Total students enrolled ISCED 5-7','Total graduates ISCED 5-7','Total students enrolled at ISCED 8',
                       'Total graduates at ISCED 8',                       
                       'Total academic staff (FTE)','Total academic staff (HC)', 
       'Number of non-academic  staff (FTE)','Number of non-academic staff (HC)',
        'Total Current expenditure (EURO)',
      'Total Current revenues (EURO)']
       

#________________________________________________________________________________


#________________________________________________________________________________

#Variables considered in the Donors Selection
variabili_per_imputati = ['Smooth Students Enrolled',
       'Smooth Students Graduates', 'Smooth PhD Enrolled', 'Smooth PhD Graduates', 'Smooth Academic Staff FTE',
       'Smooth Academic Staff HC', 'Smooth Non Academic Staff FTE',
       'Smooth Non Academic Staff HC', 'Smooth Expenditure (EURO)',
       'Smooth Revenues (EURO)']
                     
#________________________________________________________________________________

 
#________________________________________________________________________________

# Correlation Variables

variabili_correlazione = ['Total students enrolled ISCED 5-7','Total graduates ISCED 5-7','Total students enrolled at ISCED 8',
                       'Total graduates at ISCED 8',                       
                       'Total academic staff (FTE)','Total academic staff (HC)', 
       'Number of non-academic  staff (FTE)','Number of non-academic staff (HC)',
        'Total Current expenditure (EURO)',
       'Total Current revenues (EURO)' ]

#________________________________________________________________________________

# =============================================================================
# Selection of DONOR Institutions
# =============================================================================

#DONORS with valid values in all variables considered.

values_good_donor = 5

replace = ["a", "x", "xc", "xr", "nc", "c", "s", "Null"]

#Create a dictionary with all the variables that we can donate
donatori_diz_poss = {"completi":[], "non_completi":[]}

for i in set(imputation_test["ETER ID"]):
    
    #The Flag "buono" counts the number of variables a donor can donate.
    buono = 0
    
    for var in variabili_imputazione:
        valori = list(imputation_test[imputation_test["ETER ID"] == i][var])
        if any(i in valori for i in replace) == False:
            
            #Add Check for the number of 0 missing values
            if Counter(valori)["m"] == 0 and (len(valori) == 6 or len(valori) == 7):
                buono += 1
    
    if buono == len(variabili_imputazione):
        donatori_diz_poss["completi"].append(i)
     #Specify number of variables to consider an Institution as "good" Donor in the list "non_completi"   
    if buono > values_good_donor:
        #At least 5 variables completed
        donatori_diz_poss["non_completi"].append(i)


# #### Selection Of Possible Donor


#Check to maintain the number of times an Institution will be selected as Donor
#Choose the typology of Institution that can be Donor
        
donatore_scelto = { nn:0 for nn in donatori_diz_poss["non_completi"]}

#For the moment we work with Institutions not complete in all the variables.

donatori_completi = donatori_diz_poss["non_completi"]

print("Total Number of Donor selected: ")
print(len(donatori_completi))


# ## Cleaning Donor

maximum_variation = 0.4

def pulizia_donatori_buoni (dataset, variazione_max = maximum_variation):
    
    #The flag "contagio" detects Donor with strange values, so we do not consider them 
    #possible Donor
    contagio = False
    for col in dataset.columns:
        valori = dataset[col].values
        #print(valori)
        for i in range(len(valori)-1):
            #print(i)
            valore_dopo = valori[i]
            valore_prima = valori[i+1]
            if valore_dopo >= valore_prima - valore_prima*variazione_max and valore_dopo <= valore_prima + valore_prima*variazione_max :
                pass
            else:
                #print(valore_prima - valore_prima*variazione_max )
                #print(valore_prima + valore_prima*variazione_max )

                contagio = True

        #print()
        
    return contagio



def selezioneVar(data, variabili):

    miss_t = ["m","a", "x", "xc", "xr", "nc", "c", "s", "Null"]
    var_to_work = []
    for cc in variabili:
        valori = Counter(data[cc])
        buono = True
        for elem in miss_t:
            if valori[elem] != 0:
                buono = False
                break
        if buono == True:
            var_to_work.append(cc)
        
    return var_to_work


# We need a function to clean the Donor Insitutions, observing the change year after 
# year to check Institutions with an increase bigger than 50%, we do not consider as 
# "good" Donor these Institutions

donatori= []

#len(donatori)
for i in range(len(donatori_completi)):
    
    virus = imputation_test[imputation_test["ETER ID"] == donatori_completi[i]]
    #print("Working with:" + str(donatori_completi[i]))

    var_selec = selezioneVar(virus, variabili_imputazione)
    
    visrus_a = virus[var_selec]
    
    if pulizia_donatori_buoni(visrus_a) == False:
        #print(i)
        #print(donatori_completi[i])
        #print("ok")
        #print()
        donatori.append(donatori_completi[i]) 


#In this second cleaning part we detect some possible extreme values for the Ratio
#Ratio Analysis

def estrazione_valori_ratio(var):
    final = []
    for i in imputation_test.index:
        alBano = imputation_test.loc[i][var]
        if alBano != "Nada":
            final.append(alBano)
    return final

def delete_max_min(lista_valori, num_max, num_min):
    
    valori_esclusi = []
    
    for i in range(num_max):
        massimo = max(lista_valori)
        valori_esclusi.append(massimo)
        lista_valori.remove(massimo)
    for j in range(num_min):
        minimo = min(lista_valori)
        valori_esclusi.append(minimo)
        lista_valori.remove(minimo)
        
    return valori_esclusi

def exclude_eterid(var_to_work, lista_valori_da_escludere):
    eter_id_to_exclude = []

    for i in imputation_test.index:

        val_to_check = imputation_test.loc[i][var_to_work]
        if val_to_check in lista_valori_da_escludere:
            eter_id_to_exclude.append(imputation_test.loc[i]["ETER ID"])
            
    return eter_id_to_exclude


diz_reclean = {'Ratio Spese/Iscritti':[100,50], 'Ratio Academic Staff (FTE)/Iscritti':[50,100],
               'Ratio Laureati/Iscritti':[50,157],'Ratio Spese/Laureati':[50,50],'Ratio Spese/Ricavi':[50,50],
              'Ratio Non Academic Staff (FTE)/Iscritti':[20,50],
              "Ratio PhD Laureati/ PhD Iscritti":[30,30], "Ratio PhD Student/ Graduates":[30,30]
} 

for var_trial in diz_reclean:
    values_to_analyze = estrazione_valori_ratio(var_trial)  
    val_to_del= delete_max_min(values_to_analyze, diz_reclean[var_trial][0], diz_reclean[var_trial][1])
    eter_id_to_del = exclude_eterid(var_trial, val_to_del)
    
    for etid in  eter_id_to_del:
        if etid in donatori :
            donatori.remove(etid)


print("New number of Donor Institutions after the second cleaning part: ")
print(len(donatori))



# =============================================================================
# Selection of IMPUTED Institutions
# =============================================================================

#Function to extract ETER ID to be imputed.
#Create a dictionary, for each key(ETER ID) we observe which variables have missing values,
#with a number of missing equals to number of observation - 1 or a full sequence of missing.


def estrazioneImputati(dataset, variabili_riempire):
    
    diz_eterid_var_mancanti = { univ:[] for univ in set(dataset["ETER ID"])}

    for et_id in tqdm(set(dataset["ETER ID"])):

        data_lavorazione = dataset[dataset["ETER ID"] == et_id].copy()

        #Cycle on the list of variables considered in the imputation
        for vv in variabili_riempire:
   
            numero_missing = Counter(data_lavorazione[vv])["m"]
            
            #Semi-complete or full sequence of Missing
            if numero_missing >= data_lavorazione.shape[0] -1 and numero_missing != 0:

                diz_eterid_var_mancanti[et_id].append(vv)  
    
    return diz_eterid_var_mancanti

#The list of variables is "variabili_per:imputati" defined above.
diz_eterid_var_mancanti = estrazioneImputati(imputation_test, variabili_per_imputati)




#For each Institutions we know which are the missing variables, so we can consider specific
# imputed Institutions.

#We can specify the threshold according to the limit into the diz_eterid_da_imputare, to 
#select the number of Imputed Institutions.

#ETER ID with all missing variables
#diz_eterid_da_imputare = [elem for elem in diz_eterid_var_mancanti if len(diz_eterid_var_mancanti[elem]) == len(variabili_per_imputati)]

#Just some Institution
th_numb_missing_variables = 2
diz_eterid_da_imputare = [elem for elem in diz_eterid_var_mancanti if len(diz_eterid_var_mancanti[elem]) >= th_numb_missing_variables]

print("Institutions to be Imputed")
print(len(diz_eterid_da_imputare))


# =============================================================================
# WINDOW Analysis - Check used during the first part of DONOR Imputation
# =============================================================================

#Categorical Window Analysis

def finestraCategorica(dataset, lista_var_categoriche, donatori, imputato):
    
    data_work = dataset[dataset["ETER ID"] == imputato].copy()
    diz_finestra = {}
    
    for var in lista_var_categoriche:
        #We take the first value for the categorical feature cause should be the 
        #same for each year.
        diz_finestra[var] = list(data_work[var])[0]
    #print(diz_finestra)
    
    #"donatori" has to be a list.
    data_donatori = dataset[dataset["ETER ID"].isin(donatori)]
    
    for var_controllo in lista_var_categoriche:
        data_donatori = data_donatori[data_donatori[var_controllo] == diz_finestra[var_controllo]]
        
    donatori_finali = list(set(data_donatori["ETER ID"]))
    
    return donatori_finali       
        

#Here we specify the list of variables for the categorical window.    
var_categoriche_check = ["Institution Category standardized","Country Code"]



### Size Window Analysis

def calcoloMedia (lista):
    #print("Starting List: ")
    #print(lista)
    miss = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m", 'eliminare', "Nada",""]  
    
    tot = 0
    elementi = 0
    for e in lista:
        if e not in miss:
            tot += e
            elementi += 1
    if tot != 0:
        return tot/elementi
    else:
        return "Nada"


th_size_window = 0.4

def finestraSize(dataset, lista_var_size, donatori, imputato, valore_modifica_confini = th_size_window):
    #dictionary to add all the distance between the imputed Institutions and possible Donor
    diz_size_id_val = {}
    
    donatori_vincenti = []
    
    data_work = dataset[dataset["ETER ID"] == imputato].copy()
    diz_finestra = {}

    for var in lista_var_size:

        
        valore_appendere = calcoloMedia(list(data_work[var]))
    
        if valore_appendere != "Nada": 
            diz_finestra[var] = valore_appendere
    
    data_donatori = dataset[dataset["ETER ID"].isin(donatori)]
    
    #print()
    #print(diz_finestra)
    
    for possibile_donatore in set(data_donatori["ETER ID"]):
        
        #ETER ID --> Donor Institution
        diz_size_id_val[possibile_donatore] = {}

        
        conteggio = 0
        
        for vv in diz_finestra:
           
            confine = diz_finestra[vv]
            
            #Check to be sure to increase the window
            if confine*valore_modifica_confini< 10:

                finestrella = [confine - 100, confine + 100]
                #print("Modification Window")
                #print(confine)
                #print(finestrella)
                
            else:
                
                finestrella = [confine - confine*valore_modifica_confini, confine + confine*valore_modifica_confini]                

            valore_check = calcoloMedia(data_donatori[data_donatori["ETER ID"] == possibile_donatore][vv].values)
            
            try:
                diz_size_id_val[possibile_donatore][vv] = abs(valore_check - confine)/max(abs(valore_check),abs(confine))
            except:
                diz_size_id_val[possibile_donatore][vv] = "Nada"
                

            if valore_check != "Nada":
                
                #If the values falls within the window the counter is increased of 1
                if valore_check <= finestrella[1] and valore_check >= finestrella[0]:

                    conteggio += 1
                    
        #if conteggio == len(diz_finestra):
        
        if len(diz_finestra) > 0:
            
            
            #Here we specify the number of variables to consider the Institution Good 
            #to skip the first control.
            
            if conteggio >= 1:

                donatori_vincenti.append(possibile_donatore)
        else:
            donatori_vincenti.append(possibile_donatore)
            
    
    return donatori_vincenti , diz_size_id_val 




from sklearn.linear_model import LinearRegression

### Trend Window Analysis

#Compute angular coefficient (Slope)
def calcoloCoeffAngolare (lista, anni):
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
            model = LinearRegression().fit(x, y)
            return (model.coef_[0]/(sum(lista_f)/len(lista_f)))[0]    
        except:
            return "Nada"
    else:
        return "Nada"
    




th_trend_window = 0.4

def finestraTrend(dataset, lista_var_trend, donatori, imputato,valore_modifica_confini = th_trend_window):
    
    diz_trend_id_val = {}
    
    donatori_vincenti = []
    
    data_work = dataset[dataset["ETER ID"] == imputato].copy()
    diz_finestra = {}
    
    for var in lista_var_trend:
        
        valore_appendere = calcoloCoeffAngolare(data_work[var].values, data_work["Reference year"].values)
        
        if valore_appendere != "Nada": 
            diz_finestra[var] = valore_appendere
    
    
    data_donatori = dataset[dataset["ETER ID"].isin(donatori)]
    #print()
    #print(diz_finestra)
    
    for possibile_donatore in set(data_donatori["ETER ID"]):
        
        
        diz_trend_id_val[possibile_donatore] = {}
        
        conteggio = 0
        
        for vv in diz_finestra:
           
            confine = diz_finestra[vv]
            
           
            #Increase the window of the analysis 
            if confine*valore_modifica_confini < 0.05:
                
                finestrella = [confine - 0.05, confine + 0.05]
                #print("Modification Window")
                #print(confine)
                #print(finestrella)
                
            else:
                
                finestrella = [confine - confine*valore_modifica_confini, confine + confine*valore_modifica_confini]


            valore_check = calcoloCoeffAngolare(data_donatori[data_donatori["ETER ID"] == possibile_donatore][vv].values,
                                               data_donatori[data_donatori["ETER ID"] == possibile_donatore]["Reference year"].values)

            
            #Trend analysis
            try:
                diz_trend_id_val[possibile_donatore][vv] = abs(valore_check - confine)/max(abs(valore_check),abs(confine))
            except:
                diz_trend_id_val[possibile_donatore][vv] = "Nada"
            
            
            if valore_check!= "Nada":
                if valore_check <= finestrella[1] and valore_check >= finestrella[0]:

                    conteggio += 1
                    
        if len(diz_finestra) > 0:
            
            if conteggio >= 1:

                donatori_vincenti.append(possibile_donatore)
        else:
            donatori_vincenti.append(possibile_donatore)
    
    return donatori_vincenti, diz_trend_id_val     
        


# Ratio Window 
colonne_rapporto = [ i for i in imputation_test.columns if "Ratio" in i]

#print("Considered columns for the Ratio: ")
#print(colonne_rapporto)
print()


th_ratio_window = 0.4

def finestraRatio(dataset, lista_var_trend, donatori, imputato, valore_modifica_confini = th_ratio_window):
   
    donatori_vincenti = []
    
    data_work = dataset[dataset["ETER ID"] == imputato].copy()
    diz_finestra = {}

    for var in lista_var_trend:

                
        valore_appendere = calcoloMedia(list(data_work[var]))
    
        if valore_appendere != "Nada": 
            diz_finestra[var] = valore_appendere
    
    data_donatori = dataset[dataset["ETER ID"].isin(donatori)]
    
    #print()
    #print(diz_finestra)
    
    for possibile_donatore in set(data_donatori["ETER ID"]):
        
        conteggio = 0
        
        for vv in diz_finestra:
           
            confine = diz_finestra[vv]
            
            #Check to be sure to increase the Window 
            if confine*valore_modifica_confini < 0.05:
                
                
                finestrella = [confine - 0.05, confine + 0.05]
                #print("Modification Window")
                #print(confine)
                #print(finestrella)                
            else:
                
                finestrella = [confine - confine*valore_modifica_confini, confine + confine*valore_modifica_confini]


            valore_check = calcoloMedia(data_donatori[data_donatori["ETER ID"] == possibile_donatore][vv].values)

            if valore_check!= "Nada":
                if valore_check <= finestrella[1] and valore_check >= finestrella[0]:

                    conteggio += 1
                    
        #if conteggio == len(diz_finestra):
        
        if len(diz_finestra) > 0:
            
            
            if conteggio >= 1:

                donatori_vincenti.append(possibile_donatore)
        else:
            donatori_vincenti.append(possibile_donatore)
            
    
    return donatori_vincenti       
        




diz_accoppiamento_imputato_donatori = {}

#Example:
#print(diz_eterid_da_imputare = ["DE0256"])

#Main part Window Filter Analysis

diz_final_id_size = {}
diz_final_id_trend = {}

print()
print("Identification of feasible Donors")
print()

for univ_imputata in tqdm(diz_eterid_da_imputare):
    
    #print("Working with " + str(univ_imputata))
    #print(len(donatori))
  
    # 1 - Window for the Categorical filter
    
    donatori_prima_finestra = finestraCategorica(imputation_test, var_categoriche_check, donatori, univ_imputata )
    #print(len(donatori_prima_finestra))
    
    
    #Check about the number of donor Institutions we find out
    
    if len(donatori_prima_finestra) == 0:
        #print("No Donor respects the CATEGORICAL Window")
        #print()
        #print()
        pass
    else:
        #print("Go On - SIZE Window")
        
        # 2 - Window for the size filter
        donatori_seconda_finestra, diz_size_id_val = finestraSize(imputation_test, variabili_correlazione, donatori_prima_finestra, 
                                                 univ_imputata )

        #print(len(donatori_seconda_finestra))
        #print()
        
        diz_final_id_size[univ_imputata] = diz_size_id_val
        #print(diz_size_id_val)
        #print()
        
        #Check about the number of donor Institutions we find out.
        if len(donatori_seconda_finestra) == 0:
            #print("No Donor respects the SIZE Window")
            #print()
            #print()
        
            pass
        else:
            #print("Go On - TREND Window")
            
            
            # 3 - Window for the Trend filter
            
            donatori_terza_finestra, diz_trend_id_val = finestraTrend(imputation_test, variabili_correlazione, donatori_seconda_finestra,
                                                   univ_imputata )
            
            #print(len(donatori_terza_finestra))
            
            diz_final_id_trend[univ_imputata] = diz_trend_id_val
            #print(diz_trend_id_val)
            #print()
            
            if len(donatori_terza_finestra) == 0:
                
                #print("No Donor respects the TREND Window")
                #print()
                #print()
                pass
                
            else:
                
            
                # 4 - Window for the Ratios filter
                
                #print("Go On - RATIOS Window")
                
                donatori_quarta_finestra = finestraRatio(imputation_test, colonne_rapporto, donatori_terza_finestra,
                                                   univ_imputata )
            
                #print(len(donatori_quarta_finestra))
                #print()
                #print()
                
                
                diz_accoppiamento_imputato_donatori[univ_imputata] = donatori_quarta_finestra  
   



#Result Example
#print(diz_accoppiamento_imputato_donatori['DE0256'])



# =============================================================================
# # DISTANCE (K - Nearest Neighbor ) - Code
# =============================================================================

def calcoloNumeroMissing(ist_imputare, nome_inst, var_original = variabili_per_imputati,
                         var_imputati = variabili_imputazione):
    
    
    missing = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m"]
    
    #Dictionary for the distance
    
    diz_confronto_distance = {}
    for i in range(len(var_original)):
        diz_confronto_distance[var_original[i]]= var_imputati[i]
        
    #print(diz_confronto_distance)
    
    original = imputation_test[imputation_test["ETER ID"] == ist_imputare]
    check = imputation_test[imputation_test["ETER ID"] == nome_inst]
    
    #The count(conto) is tracking the number of variables we will impute.
    #We do not consider "Size" ,"Trend" and Num_Val_Selected
    
    conto = 0
    
    var_to_check = []
    for vv in diz_confronto_distance:
        
        #Istitutions (ETER ID) to impute
        valori_orig = original[vv].values
        
        risp_original =  any(item in valori_orig for item in missing)
        
        if risp_original == True:
            var_to_check.append(vv)
            
         #Istitution possible donor
    #print(var_to_check)
    
    for vv in var_to_check:
        
        #print(vv)
        #print(diz_confronto_distance[vv])

        valori = check[diz_confronto_distance[vv]].values

        risp =  any(item in valori for item in missing)

        if risp == False:
            conto += 1
            
    #print()
    #print(len(var_to_check), conto)
    #print()
    val_to_return = len(var_to_check) - conto 
    
    return val_to_return



#Function for the genertion of the vector considering the values of a specific ETER ID and
#all possible Donor Institution

def creareVettore_Knn(name_imputato, lista_donatori, diz_valori_imputato, diz_size, diz_trend):
    
    #Big vector for all the distances
    distanze = []
    missing = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m"]
    
    for nome in lista_donatori:
        #print("Working with " + nome)
        
        indice_donatore = imputation_test[imputation_test["ETER ID"] == nome].index[0]
        
        distanza_singola_univ = []
        
        for var in diz_valori_imputato:
            
            #print(var)
            #print()
            
            if var == "Size":
                
                distanza_singola_univ.append(calcoloMaxVariazione(diz_size[nome]))
                
                #print("SIZE")
                #print(calcoloMaxVariazione(diz_size[nome]))
                
            
            if var == "Trend":
                
                distanza_singola_univ.append(calcoloMaxVariazione(diz_trend[nome]))
                
                
                #print("TREND")
                #print(calcoloMaxVariazione(diz_trend[nome]))
                
            if var == "Num_Val_Selected":
                
                #This function compute the number of variables the Donor can give to the 
                #Institution imputed
                distanza_singola_univ.append(calcoloNumeroMissing(name_imputato,nome))

            
            if var == "Institution Category standardized" :
                
                valore_donat = imputation_test.loc[indice_donatore]["Institution Category standardized"]
                #print(valore_donat)
                #print(diz_valori_imputato[var])

                if  valore_donat == diz_valori_imputato[var]:
                    distanza_singola_univ.append(0)
                elif valore_donat in missing:
                    #distanza_singola_univ.append(np.NaN)
                    distanza_singola_univ.append(3)
                else:
                    distanza_singola_univ.append(3)
                
            
            if var == "Institution Category - English" :
                
                valore_donat = imputation_test.loc[indice_donatore]["Institution Category - English"]
                
                #print(valore_donat)
                #print(diz_valori_imputato[var])
                
                if  valore_donat == diz_valori_imputato[var]:
                    distanza_singola_univ.append(0)
                elif valore_donat in missing:
                    distanza_singola_univ.append(3)
                    #distanza_singola_univ.append(np.NaN)
                else:
                    distanza_singola_univ.append(3)

            if  var == "Distance education institution":
                
                valore_donat = imputation_test.loc[indice_donatore]["Distance education institution"]
                
                #print(valore_donat)
                #print(diz_valori_imputato[var])

                if valore_donat == diz_valori_imputato[var]:
                    distanza_singola_univ.append(0)
                elif valore_donat in missing:
                    distanza_singola_univ.append(3)
                    #distanza_singola_univ.append(np.NaN)
                else:
                    distanza_singola_univ.append(3)

            if var == "Legal status" :
                
                valore_donat = imputation_test.loc[indice_donatore]["Legal status"]
                
                #print(valore_donat)
                #print(diz_valori_imputato[var])

                if valore_donat == diz_valori_imputato[var]:
                    distanza_singola_univ.append(0)
                elif valore_donat in missing:
                    distanza_singola_univ.append(3)
                    #distanza_singola_univ.append(np.NaN)
                else:
                    distanza_singola_univ.append(3)

            if var == "Country Code":
                
                valore_donat = imputation_test.loc[indice_donatore]["Country Code"]
                
                #print(valore_donat)
                #print(diz_valori_imputato[var])
                
                if valore_donat == diz_valori_imputato[var]:
                    distanza_singola_univ.append(0)
                
                elif similarity_country[valore_donat] == similarity_country[diz_valori_imputato[var]]:
                    distanza_singola_univ.append(3)
                else:
                
                # In case the 2 conditions are not respected delete all with Nada value
                    
                    #distanza_singola_univ.append(np.NaN)
                    distanza_singola_univ.append(3)
                    
                    
        #print("Add all this smal vector to the final Vector for the distance computation")
        distanze.append(distanza_singola_univ)
    
    #print(distanze)
    
    #Check numero di occorrenze
    #print(len(distanze))
    
    #Here we add the vector of the Institution imputed, all 0 cause there is no distance
    distanze.insert(0, [0]*len(distanze[0]))
    #print(distanze)
            
    return distanze



max_selection_Donor = 2

def aggiornamentoDonatori(donatori, diz_scelta_donatori, last_donor_select):
    #print("Possible Donors before the Check: " + str(len(donatori)))
    
    #Below we add a constraint to the number of times a Donor can be selected (max_selection_Donor)
    scelti = [val_id for val_id in diz_scelta_donatori if diz_scelta_donatori[val_id] == max_selection_Donor]
    
    for elem in scelti:
        if elem in donatori:
            donatori.remove(elem)
    
    
    #Check based on the last Donor selected
    if last_donor_select != None and last_donor_select!= "Nada":
        
        #Check if really there is a Donor
        if last_donor_select in donatori:
            donatori.remove(last_donor_select)
        
    #print("Possible Donors after the Check: " + str(len(donatori)))
        
    return donatori
    




#Variables used to compute the KNN Distance
variabili_KNN = ["Institution Category standardized", "Institution Category - English", 
                 "Country Code", "Distance education institution", "Legal status" ]




def calcoloDistanza(dataset, variabili_distanza, imputato, donatori, diz_size_id_val, diz_trend_id_val):
    
    
    #print("Starting DISTANCE Computation")
    
    data_work = dataset[dataset["ETER ID"] == imputato].copy()
    
    #Here we check the number of Donor, at the moment they should be at least 1
    if len(donatori) >= 1:
        
        #print("DONOR analyzed: " + str(len(donatori)))

        # Now we compute the nearest Donor according to KNN

     
        miss = ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m", "Nada"] 

        var_finali = {}

        indice_imputato = data_work.index[0]
        #print("Selected Index " + str(indice_imputato))

        for considerate in variabili_distanza:
            #print(considerate)
            valore = data_work.loc[indice_imputato][considerate]
            #print(valore)

            if valore not in miss:
                var_finali[considerate] = valore

    
        
        #Add the SIZE for the Distance Computation
        for elem in diz_size_id_val:
            if len(diz_size_id_val[elem]) > 0:
                var_finali["Size"] = 0

            break
            
        
        #Add the TREND for the Distance Computation
        for elem in diz_trend_id_val:
            if len(diz_trend_id_val[elem]) > 0:
                var_finali["Trend"] = 0

                break
                
        
        ##Add the Num_Val_Selected for the Distance Computation:
        var_finali["Num_Val_Selected"] = 0
        
        
        #Now we create the Distance Matrix considering all the possible Donor Institutions
        
        print("Variables available for DISTANCE Computation")
        print(var_finali)
          
        distanze_KNN = creareVettore_Knn(imputato, donatori, var_finali, diz_size_id_val, diz_trend_id_val)
        
        #print(distanze_KNN)

        #We delete the NaN values
        distanze_KNN = np.where(np.isnan(distanze_KNN), 1000000000, distanze_KNN)

        X = np.array(distanze_KNN)

        #Here we have k=2 because the nearest should be always itself
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', n_jobs = -1).fit(X)
        distances, indices = nbrs.kneighbors(X)

        #print(distances)
        #print(indices)
        
        #Last check about the correctness of the Donor, that should not be itself.
        
        if indices[0][1] != 0:
            #print(donatori[indices[0][1] -1])
            donar = donatori[indices[0][1] -1]
            distanza_finale = distances[0][1]
        else:
            #print(donatori[indices[0][0] -1])
            donar = donatori[indices[0][0] -1]
            distanza_finale = distances[0][0]

        #print("Mapping Imputed-Donor:")
        #print(name_univ )
        #print(donar)
        
        return donar, distanza_finale, len(donatori)
    
    else:
        print("NO Variables available for DISTANCE Computation")
        #print(donatori)
        return "Nada", "Nada", len(donatori)





def calcoloMaxVariazione(dizionario):
    #print(dizionario)
    #valori_analisi = [elem for elem in list(dizionario.values()) if str(int(elem)).isnumeric()]
    valori_analisi = []
    for elem in list(dizionario.values()):
        #print(elem)
        if elem != "Nada" and not math.isinf(elem) and not np.isnan(elem):
            
            #We need to work with the ABS value of the TREND,
            #otherwise we got error

            if str(int(abs(elem))).isnumeric():
                #Append Maximum values in an absolute way
                valori_analisi.append(abs(elem))
                
    massimo = max(valori_analisi)
    return massimo*3





#MAIN DISTANCE - Computation

risultato_accoppiamento = {}
risultato_distanze= {}




#We follow the order of Country Code established at the beginning
for naz_selected in priority_for_donor_imputation:
    #print(naz_selected)
    
    possible_ETERID_nations = list(set(imputation_test[imputation_test["Country Code"] == naz_selected]["ETER ID"]))
    #print(possible_ETERID_nations)
    
    #Save the last Donor selected, to not take twice in a row
    donatore_to_delete = None
    
    for name_univ in possible_ETERID_nations:
    
        if name_univ in diz_accoppiamento_imputato_donatori:
        
            #If ETER ID is in the Dictionary mapping we will work on it
            print("Working with " + str(name_univ))

            if len(diz_accoppiamento_imputato_donatori[name_univ]) > 0 :


                #Check on the Donor Institutions we remove the ETER ID already selected
                donatori_possibili = aggiornamentoDonatori(diz_accoppiamento_imputato_donatori[name_univ], donatore_scelto, donatore_to_delete)

                
                #print()
                #print("Possible DONORS")
                #print(donatori_possibili)
                
                if name_univ in donatori_possibili:
                    #print("Remove DONOR with same ETER ID of the Imputed Institution")
                    donatori_possibili.remove(name_univ)
                    
                #print()
                #print("After first cleaning of Donors, they are: " + str(len(donatori_possibili)))
                
                # Dataset - Var Distance - ETER ID - All DONOR after Windows
                donatore_finale, dist_finale, donatori_osservati = calcoloDistanza(imputation_test, variabili_KNN, name_univ, 
                                                                                   donatori_possibili, diz_final_id_size[name_univ],
                                                                                  diz_final_id_trend[name_univ])
                
                
                
                if donatore_finale != "Nada":
                
                    #print(donatore_finale, dist_finale, donatori_osservati)
                    print("Best DONOR available:" + str(donatore_finale) + " , Distance: " + str(dist_finale))
                    
                else:
                    print("NO DONORS SUITABLE for this Institution, leaving it UNIMPUTED")
                
    
                risultato_accoppiamento[name_univ] = donatore_finale

                risultato_distanze[name_univ] = [donatore_finale, dist_finale, donatori_osservati]

                print()
                

                
                #Max number of selection of a specific Donor
                donatore_to_delete = donatore_finale
                
                if donatore_finale != 'Nada' and len(diz_eterid_per_normalizzare[name_univ]["1"]) == 0 and len(diz_eterid_per_normalizzare[name_univ]["2o+"]) == 0:
                    donatore_scelto[donatore_finale] += 1
                    #print("UPDATE +1")
                    #print(name_univ)
                    #print()
            else:
                print("NO DONORS SUITABLE for this Institution, leaving it UNIMPUTED")
                print()
                




# Creation of a dictionary with all the results MAPPING {Eter ID Imputed: ETER ID Donor_selected}

diz_eterid_univ ={}

for elem in set(imputation_test["ETER ID"]) :
    if elem not in diz_eterid_univ:
        nn = imputation_test[imputation_test["ETER ID"] == elem]["Institution Name"] .values
        diz_eterid_univ[elem] = nn[0]


#______________________________________________________________________________

        
diz_corrispondenze_variabili = {'Smooth Students Enrolled': 'Smooth Students Enrolled',
 'Smooth Students Graduates':'Smooth Students Graduates',
 'Smooth Academic Staff FTE':"Smooth Academic Staff FTE",
 'Smooth Academic Staff HC': "Smooth Academic Staff HC",
 'Smooth Non Academic Staff FTE':"Smooth Non Academic Staff FTE",
 'Smooth Non Academic Staff HC':"Smooth Non Academic Staff HC",
 'Smooth Expenditure (EURO)':"Smooth Expenditure (EURO)",
 'Smooth Revenues (EURO)':"Smooth Revenues (EURO)",
                   "Smooth PhD Enrolled": "Smooth PhD Enrolled" ,
                    "Smooth PhD Graduates": "Smooth PhD Graduates"}




#NORMALIZATION: Here we observe available value within the Institution to normalize
# the imputed variable according to the other available values.

def normalization_method(imp, don, index_imp, index_don, var_da_imputare, diz_corrispondenze_variabili = diz_corrispondenze_variabili):
    
    
    
    diz_inverse_typology = {}
    for k in diz_eterid_per_normalizzare[imp]:
        for elem in diz_eterid_per_normalizzare[imp][k]:
            diz_inverse_typology[elem] = k
    
    
    if var_da_imputare in diz_eterid_per_normalizzare[imp]["1"]:
        #print(1)
        
        
        
        check = imputation_test[imputation_test["ETER ID"] == imp].copy()
        
        #Years and Values
        years = check["Reference year"].values
        val_old = check[var_da_imputare].values
        
        dict_values = dict(zip(years,val_old))
        
        for i in dict_values:
            if dict_values[i] != "m":
                #print(dict_values[i])
                #print(imputation_test[(imputation_test["ETER ID"] == don) & (imputation_test["Reference year"] == i)][var_da_imputare].values[0])
                
                try:

                    ratio_compute =  dict_values[i]/imputation_test[(imputation_test["ETER ID"] == don) & (imputation_test["Reference year"] == i)][var_da_imputare].values[0]
                    #print(ratio_compute)
                    new_val_add = (imputation_test.loc[index_don][var_da_imputare])*ratio_compute
                    #print(float(new_val_add))
                    #print("NORMALIZED")
                    
                    return float(new_val_add)
                except:
                    #print(i)
                    #print("Value generates Error")
                    
                    #return "2017"
                    return "No"
                    

    elif var_da_imputare in diz_eterid_per_normalizzare[imp]["2o+"]:
        
        #print(2)

        check = imputation_test[imputation_test["ETER ID"] == imp].copy()

        #Years and Values
        years = check["Reference year"].values
        val_old = check[var_da_imputare].values

        dict_values = dict(zip(years,val_old))

        for i in dict_values:
            if dict_values[i] != "m":
                #print(dict_values[i])
                #print(imputation_test[(imputation_test["ETER ID"] == don) & (imputation_test["Reference year"] == i)][var_da_imputare].values[0])

                try:

                    ratio_compute =  dict_values[i]/imputation_test[(imputation_test["ETER ID"] == don) & (imputation_test["Reference year"] == i)][var_da_imputare].values[0]
                    #print(ratio_compute)
                    new_val_add = (imputation_test.loc[index_don][var_da_imputare])*ratio_compute
                    #print(float(new_val_add))
                    #print("NORMALIZED")

                    return float(new_val_add)
                except:
                    #print(i)
                    #print("Value generates Error")

                    #return "2017"
                    return "No"
                    
    
    elif diz_corrispondenze_variabili[var_da_imputare] in diz_eterid_per_normalizzare[imp]["2o+"] + diz_eterid_per_normalizzare[imp]["1"]:
        
        #print(3)
        
        var_choose_norm = diz_corrispondenze_variabili[var_da_imputare]
        #print(var_choose_norm)
        
        val_select = diz_inverse_typology[var_choose_norm]
        
        if val_select == "1":
            
            check = imputation_test[imputation_test["ETER ID"] == imp].copy()

            #Years and Values
            years = check["Reference year"].values
            val_old = check[var_choose_norm].values

            dict_values = dict(zip(years,val_old))

            for i in dict_values:
                if dict_values[i] != "m":
                    #print(dict_values[i])
                    #print(imputation_test[(imputation_test["ETER ID"] == don) & (imputation_test["Reference year"] == i)][var_choose_norm].values[0])

                    ratio_compute =  dict_values[i]/imputation_test[(imputation_test["ETER ID"] == don) & (imputation_test["Reference year"] == i)][var_choose_norm].values[0]
                    #print(ratio_compute)
                    new_val_add = (imputation_test.loc[index_don][var_da_imputare])*ratio_compute

                    #print("NORMALIZED")

                    return float(new_val_add)
                    
            
        elif val_select == "2o+":
            
            #print(4)
        
            #Compute Ratio
            ratio_compute = imputation_test.loc[index_imp][var_choose_norm] / imputation_test.loc[index_don][var_choose_norm]

            #print(ratio_compute)
            #New value to add
            new_val_add = imputation_test.loc[index_don][var_da_imputare]*ratio_compute
            #print(imputation_test.loc[index_don][var_da_imputare])
            #print(round(new_val_add),0)
            #print("NORMALIZED")

            return float(new_val_add)
        
        
    elif len(diz_eterid_per_normalizzare[imp]["2o+"])>=1 or len(diz_eterid_per_normalizzare[imp]["1"])>=1:
        
        #print(5)
        
        if len(diz_eterid_per_normalizzare[imp]["2o+"])>=1:
            
            
            var_choose_norm = diz_eterid_per_normalizzare[imp]["2o+"][0]
            
            #print(var_choose_norm)
            
            #Compute Ratio
            ratio_compute = imputation_test.loc[index_imp][var_choose_norm] / imputation_test.loc[index_don][var_choose_norm]

            #print(ratio_compute)
            #New value to add
            new_val_add = imputation_test.loc[index_don][var_da_imputare]*ratio_compute
            #print(imputation_test.loc[index_don][var_da_imputare])
            #print(round(new_val_add),0)
            #print("NORMALIZED")

            return float(new_val_add)
            
        elif len(diz_eterid_per_normalizzare[imp]["1"])>=1:
            
            var_choose_norm = diz_eterid_per_normalizzare[imp]["1"][0]
            
            #print(var_choose_norm)
            
            check = imputation_test[imputation_test["ETER ID"] == imp].copy()

            #Years and Values
            years = check["Reference year"].values
            val_old = check[var_choose_norm].values

            dict_values = dict(zip(years,val_old))

            for i in dict_values:
                if dict_values[i] != "m":
                    #print(dict_values[i])
                    #print(imputation_test[(imputation_test["ETER ID"] == don) & (imputation_test["Reference year"] == i)][var_choose_norm].values[0])

                    ratio_compute =  dict_values[i]/imputation_test[(imputation_test["ETER ID"] == don) & (imputation_test["Reference year"] == i)][var_choose_norm].values[0]
                    #print(ratio_compute)
                    
                    if type(imputation_test.loc[index_don][var_da_imputare]) != str:
                        #print(imputation_test.loc[index_don][var_da_imputare])
                        new_val_add = (imputation_test.loc[index_don][var_da_imputare])*ratio_compute
                        new_val_add = float(new_val_add)
                        
                    else:
                        new_val_add = "No"

                    #print("NORMALIZED")

                    return new_val_add  
        
    else:
        #print("NOT NORMALIZED")
        return "No"




combinazione_var = {'Smooth Students Enrolled': 'Enrolled',
 'Smooth Students Graduates':'Graduates',
 'Smooth Academic Staff FTE':"Academic Staff (FTE)",
 'Smooth Academic Staff HC': "Academic Staff (HC)",
 'Smooth Non Academic Staff FTE':"Non Academic Staff (FTE)",
 'Smooth Non Academic Staff HC':"Non Academic Staff (HC)",
 'Smooth Expenditure (EURO)':"Expenditure (EURO)",
 'Smooth Revenues (EURO)':"Revenues (EURO)",
                   "Smooth PhD Enrolled": "PhD Enrolled" ,
                    "Smooth PhD Graduates": "PhD Graduates"}




#ADD ALL THE RESULTS, considering ETER ID Donor, Value , Distance and Flag(definition of Donor)

#Variables taken from the dictionary
print("Add Donor Imputation Results")
print()
for possible in tqdm(combinazione_var):
    #print("Working with")
    #print(possible)
    
    variabile_imputata_con_donatore = possible
    estensione = combinazione_var[possible]

    donor = []
    distanza = []
    val_imputato = []
    flag = []

    nome_donor = []

    for i in imputation_test.index:

        chiave = imputation_test.loc[i]["ETER ID"]
        stop = imputation_test.loc[i][variabile_imputata_con_donatore]


        #Verify imputed value is "m"
        if stop == "m":
            
            #print("Value with Missing")

            if chiave in risultato_accoppiamento :

                #print(chiave)

                if risultato_accoppiamento[chiave] != 'Nada':
                    
                    #print("Missing with Donor different from Nada")

                    anno = imputation_test.loc[i]["Reference year"]
                    
                    
                    try:

                        #Extract Index of Donor
                        indice_donatore = imputation_test[(imputation_test["ETER ID"] == risultato_accoppiamento[chiave]) & (imputation_test["Reference year"] == anno)].index[0]


                        #print("Starting Normalization")
                        #Compute possible Normalization
                        valore_da_aggiungere = normalization_method(chiave, risultato_accoppiamento[chiave], i, indice_donatore, variabile_imputata_con_donatore)

                        if valore_da_aggiungere != "No" and valore_da_aggiungere != "2017":
                            
                            donor.append(diz_eterid_univ[risultato_accoppiamento[chiave]])

                            nome_donor.append(risultato_accoppiamento[chiave])

                            distanza.append(risultato_distanze[chiave][1])

                            val_imputato.append(valore_da_aggiungere)

                            flag.append("Donor Rescaled")

                        elif valore_da_aggiungere == "No":
                            
                            if imputation_test[(imputation_test["ETER ID"] == risultato_accoppiamento[chiave]) & (imputation_test["Reference year"] == anno)][variabile_imputata_con_donatore].values[0] in ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m"] :
                            
                                donor.append("")

                                nome_donor.append("")

                                distanza.append("")

                                val_imputato.append(imputation_test[(imputation_test["ETER ID"] == chiave) & (imputation_test["Reference year"] == anno)][variabile_imputata_con_donatore].values[0])
                                flag.append("") 
                            
                            else:
                            
                                donor.append(diz_eterid_univ[risultato_accoppiamento[chiave]])

                                nome_donor.append(risultato_accoppiamento[chiave])

                                distanza.append(risultato_distanze[chiave][1])

                                val_imputato.append(imputation_test[(imputation_test["ETER ID"] == risultato_accoppiamento[chiave]) & (imputation_test["Reference year"] == anno)][variabile_imputata_con_donatore].values[0])
                                flag.append("Donor")
                    except:
                        #print("IMPOSSIBLE to extract Donor Index")
                        
                        donor.append("")

                        nome_donor.append("")

                        distanza.append("")

                        #print(anno)
                        val_imputato.append(imputation_test[(imputation_test["ETER ID"] == chiave) & (imputation_test["Reference year"] == anno)][variabile_imputata_con_donatore].values[0])
                        flag.append("")

                else:
                    #print("Missing with DONOR "Nada")
                    
                    #donor.append(risultato_accoppiamento[chiave])
                    donor.append("")

                    nome_donor.append("")

                    distanza.append("")

                    flag.append("")

                    anno = imputation_test.loc[i]["Reference year"]
                    
                    val_imputato.append(imputation_test[(imputation_test["ETER ID"] == chiave) & (imputation_test["Reference year"] == anno)][variabile_imputata_con_donatore].values[0])
                    
                


            else:
                
                #print("Missing WITHOUT DONOR")
                
                donor.append("")
                nome_donor.append("")
                distanza.append("")
                flag.append("")
                anno = imputation_test.loc[i]["Reference year"]
                val_imputato.append(imputation_test.loc[(imputation_test["ETER ID"] == chiave) & (imputation_test["Reference year"] == anno)][variabile_imputata_con_donatore].values[0])

        #If the value is not "m" we maintain the old value
        else:
            #print("Value without Missing")

            donor.append("")
            nome_donor.append("")
            distanza.append("")
            flag.append("")
            anno = imputation_test.loc[i]["Reference year"]
            val_imputato.append(imputation_test.loc[(imputation_test["ETER ID"] == chiave) & (imputation_test["Reference year"] == anno)][variabile_imputata_con_donatore].values[0])   
        
        
    imputation_test["Name Donor " + estensione] = donor
    imputation_test["Value Donor " + estensione] = val_imputato
    imputation_test["Flag " + estensione] = flag
    imputation_test["Distance Donor " + estensione] = distanza
    imputation_test["ID Donor " + estensione] = nome_donor




#Saving the dataset for the validation
imputation_test.to_excel(name_file_saved,index = False)

print("Donor Imputation Completed")
print()