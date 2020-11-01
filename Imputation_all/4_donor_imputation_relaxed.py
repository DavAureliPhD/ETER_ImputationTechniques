# -*- coding: utf-8 -*-
"""
@author: Davide Aureli, Renato Bruni, Cinzia Daraio

"""

# =============================================================================
# LIBRARY
# =============================================================================
import pandas as pd
from tqdm import tqdm
from collections import Counter
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

#Seed value to obtain always the same result according to the donor imputation part.
import random
random.seed(42)


# =============================================================================
# PARAMETERS - Defined here, the user can change them
# =============================================================================

#Read dataset after first part of Donor imputation
with open('path_imputation_donor.txt', encoding='utf-8') as f:
    path = f.read().replace('\n', '')

name_starting_file = path + "fileout_donor.xlsx"

#Reading the dataset
imputation_test = pd.read_excel(name_starting_file)

#directory and name of the saved file
name_file_saved = path + "fileout_donor_relaxed.xlsx"

#Research analysis
with open('research_analysis.txt', encoding='utf-8') as f:
    research_analysis = f.read().replace('\n', '')

#________________________________________________________________________________

#Define variables used for the normalization


var_normalization = ["Value Donor Enrolled", "Value Donor Graduates", "Value Donor PhD Enrolled", 
                     "Value Donor PhD Graduates", "Value Donor Academic Staff (HC)","Value Donor Expenditure (EURO)",
                        "Value Donor Revenues (EURO)", 'Value Donor Non Academic Staff (HC)', 
                         'Value Donor Non Academic Staff (FTE)', "Value Donor Academic Staff (FTE)"]
#________________________________________________________________________________

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

#For Example:
#print("Small Example of one University: \n" )
#ETER ID = IT0005
#diz_eterid_per_normalizzare['IT0005']



# =============================================================================
# Define ALL the Variables in the DONOR Imputation
# =============================================================================

# Now we define the variables on which we will work, 
# the correlation variables and finally we define the possible Donor Institutions.




#Define variable for the imputation
    
variabili_imputazione = ['Smooth Students Enrolled',
       'Smooth Students Graduates', 'Smooth PhD Enrolled', 'Smooth PhD Graduates', 'Smooth Academic Staff FTE',
       'Smooth Academic Staff HC', 'Smooth Non Academic Staff FTE',
       'Smooth Non Academic Staff HC', 'Smooth Expenditure (EURO)',
       'Smooth Revenues (EURO)'
                       ]


#________________________________________________________________________________

                     
#Variables considered in the Donors Selection

variabili_per_imputati = ["Value Donor Enrolled", "Value Donor Graduates", "Value Donor PhD Enrolled", 
                     "Value Donor PhD Graduates", "Value Donor Academic Staff (HC)","Value Donor Expenditure (EURO)",
                         "Value Donor Revenues (EURO)", 'Value Donor Non Academic Staff (HC)', 
                         'Value Donor Non Academic Staff (FTE)', "Value Donor Academic Staff (FTE)"]

#________________________________________________________________________________



#________________________________________________________________________________


# Correlation Variables

variabili_correlazione = ['Smooth Students Enrolled',
       'Smooth Students Graduates', 'Smooth PhD Enrolled', 'Smooth PhD Graduates', 'Smooth Academic Staff FTE',
       'Smooth Academic Staff HC', 'Smooth Non Academic Staff FTE',
       'Smooth Non Academic Staff HC', 'Smooth Expenditure (EURO)',
       'Smooth Revenues (EURO)'
                       ]
#________________________________________________________________________________




# =============================================================================
# Selection of DONOR Institutions
# =============================================================================

#DONATORI with valid values in all variables considered.

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






#### Selection Of Possible Donor

#Check to maintain the number of times an Institution will be selected as Donor
#Choose the typology of Institution that can be Donor
        
donatore_scelto = { nn:0 for nn in donatori_diz_poss["completi"]}

#For the moment we work with Institutions complete in all the variables.

donatori_completi = donatori_diz_poss["completi"]

print("Total Number of Donor selected: ")
print(len(donatori_completi))



# ## Cleaning Donor

maximum_variation = 0.6


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
    
var_categoriche_check = ["Institution Category standardized"]



### Size Window Analysis

def calcoloMedia (lista):
    #print("Lista iniziale: ")
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

th_size_window = 0.6

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

                #10
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

#Function of angular coefficient (Slope)
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
            #model = LinearRegression()
            #model.fit(x, y)
            model = LinearRegression().fit(x, y)
            return (model.coef_[0]/(sum(lista_f)/len(lista_f)))[0]    
        except:
            return "Nada"
    else:
        return "Nada"
    





th_trend_window = 0.6

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
        



# Ratios Window 
colonne_rapporto = [ i for i in imputation_test.columns if "Ratio" in i]

#print("Considered columns for the Ratio: ")
#print(colonne_rapporto)
print()



th_ratio_window = 0.6

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
        



### Main part Window Filter Analysis

diz_accoppiamento_imputato_donatori = {}

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
#        print("No Donor respects the CATEGORICAL Window")
#        print()
#        print()
        pass
    else:
#        print("Go On - SIZE Window")
        
        
        # 2 - Window for the size filter
        
        donatori_seconda_finestra, diz_size_id_val = finestraSize(imputation_test, variabili_correlazione, donatori_prima_finestra, 
                                                 univ_imputata )

#        print(len(donatori_seconda_finestra))
#        print()
        
        diz_final_id_size[univ_imputata] = diz_size_id_val
        #print(diz_size_id_val)
        #print()
        
        #Check about the number of donor Institutions we find out.
        if len(donatori_seconda_finestra) == 0:
#            print("No Donor respects the SIZE Window")
#            print()
#            print()
        
            pass
        else:
#            print("Go On - TREND Window")
            
            
            # 3 - Window for the Trend filter
            
            donatori_terza_finestra, diz_trend_id_val = finestraTrend(imputation_test, variabili_correlazione, donatori_seconda_finestra,
                                                   univ_imputata )
            
#            print(len(donatori_terza_finestra))
            
            diz_final_id_trend[univ_imputata] = diz_trend_id_val
            #print(diz_trend_id_val)
            #print()
            
            if len(donatori_terza_finestra) == 0:
                
#                print("No Donor respects the TREND Window")
#                print()
#                print()
                pass
            else:
#                print("Go On - RATIO Window")
            
                # 4 - Window for the Ratio filter
                
                donatori_quarta_finestra = finestraRatio(imputation_test, colonne_rapporto, donatori_terza_finestra,
                                                   univ_imputata )
            
#                print(len(donatori_quarta_finestra))
#                print()
#                print()
                
                
                diz_accoppiamento_imputato_donatori[univ_imputata] = donatori_quarta_finestra  
   



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





#Open Similarity_country dictionary
with open('similarity_country.pkl', 'rb') as handle:
    similarity_country = pickle.load(handle)

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
                    
                    
        #print("Add all this small vector to the final Vector for the distance computation")
        distanze.append(distanza_singola_univ)
    
    #print(distanze)
    
    #Check numero di occorrenze
    #print(len(distanze))
    
    #Here we add the vector of the Institution imputed, all 0 cause there is no distance
    distanze.insert(0, [0]*len(distanze[0]))
    #print(distanze)
            
    return distanze





max_selection_Donor = 15

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
    
    
    #print("Starting with DISTANCE Computation")
    
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

        #Here we have k=4 because the nearest should be always itself
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




#Re-load the ordered list of nations to impute
with open('priority_for_donor_imputation.pkl', 'rb') as handle:
    priority_for_donor_imputation = pickle.load(handle)


#Main Distance Computation

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
                #print("DONOR")
                #print(donatori_possibili)
                
                if name_univ in donatori_possibili:
                    print("Remove DONOR with same ID of the Imputed Institution")
                    donatori_possibili.remove(name_univ)
                    
                #print()
                #print("After first cleaning of Donors, they are: ")
                #print(len(donatori_possibili))
                
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


diz_corrispondenze_variabili = {'Value Donor Enrolled':'Value Donor Graduates',
                               'Value Donor Graduates':'Value Donor Enrolled',
                               'Value Donor Academic Staff (FTE)':'Value Donor Academic Staff (HC)',
                               'Value Donor Academic Staff (HC)':'Value Donor Academic Staff (FTE)',
                               'Value Donor PhD Enrolled':'Value Donor PhD Graduates',
                                'Value Donor PhD Graduates':'Value Donor PhD Enrolled',
                               'Value Donor Expenditure (EURO)':'Value Donor Revenues (EURO)',
                                'Value Donor Revenues (EURO)':'Value Donor Expenditure (EURO)',
                               "Value Donor Non Academic Staff (HC)":'Value Donor Non Academic Staff (FTE)',
                                'Value Donor Non Academic Staff (FTE)':"Value Donor Non Academic Staff (HC)"}

#______________________________________________________________________________




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
    
    
    
### Add values without replacing the FIRST imputation made by Donor Code


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




#Function to add a little noise to the imputation value between +10% and -10%

def computeAddingValue(mu):
    sigma = 0.1*mu
    s = int(np.random.normal(mu, sigma, 1))
    return s
    





#Foe each index
for i in tqdm(imputation_test.index):
  
    #for each variable imputed
    for variable_work in diz_variabili_colonne_modifiche:
        #print("Working with " + str(variable_work))

        if diz_variabili_colonne_modifiche[variable_work][4] in variabili_imputazione:
    
            chiave = imputation_test.loc[i]["ETER ID"]
            #Check about the donor value if it is "m"
            stop = imputation_test.loc[i][diz_variabili_colonne_modifiche[variable_work][5]]
            #print(stop)
            
            #Check about the value imputed if it is "m"
            if stop == "m":

                if chiave in risultato_accoppiamento :

                    #print(chiave)


                    if risultato_accoppiamento[chiave] != 'Nada': 
                        
                        anno = imputation_test.loc[i]["Reference year"]
                            
                        try:
                            #Extract Index of DONOR
                            indice_donatore = imputation_test[(imputation_test["ETER ID"] == risultato_accoppiamento[chiave]) & (imputation_test["Reference year"] == anno)].index[0]

                            #print("INDEX")
                            #print(indice_donatore)

                            #print("Starting Normalization Computation")
                            #Computaion of the possible Normalization
                            valore_da_aggiungere = normalization_method(chiave, risultato_accoppiamento[chiave], i, indice_donatore, diz_variabili_colonne_modifiche[variable_work][5])

                            if valore_da_aggiungere != "No" and valore_da_aggiungere != "2017":

                                imputation_test[diz_variabili_colonne_modifiche[variable_work][-1]][i] = computeAddingValue(valore_da_aggiungere)
                                
                                
                                
                                imputation_test[diz_variabili_colonne_modifiche[variable_work][3]][i] = "Donor Relaxed Rescaled"
                                

                                #Add name of Donor Selected 

                                imputation_test[diz_variabili_colonne_modifiche[variable_work][1]][i] = diz_eterid_univ[risultato_accoppiamento[chiave]]

                                #Add ETER ID of Donor Selected

                                imputation_test[diz_variabili_colonne_modifiche[variable_work][0]][i] = risultato_accoppiamento[chiave]


                                #Add Distance Donor selected
                                imputation_test[diz_variabili_colonne_modifiche[variable_work][2]][i] = risultato_distanze[chiave][1] 


                                
                            elif valore_da_aggiungere == "No":
                                if imputation_test[(imputation_test["ETER ID"] == risultato_accoppiamento[chiave]) & (imputation_test["Reference year"] == anno)][diz_variabili_colonne_modifiche[variable_work][5]].values[0] not in ["a", "x", "xc", "xr", "nc", "c", "s", "Null", "m"]:
                                        
                                    imputation_test[diz_variabili_colonne_modifiche[variable_work][-1]][i] = imputation_test[(imputation_test["ETER ID"] == risultato_accoppiamento[chiave]) & (imputation_test["Reference year"] == anno)][diz_variabili_colonne_modifiche[variable_work][5]].values[0]
                                
                                
                                    imputation_test[diz_variabili_colonne_modifiche[variable_work][3]][i] = "Donor Relaxed"
                                    
                                    #Add name of Donor Selected  

                                    imputation_test[diz_variabili_colonne_modifiche[variable_work][1]][i] = diz_eterid_univ[risultato_accoppiamento[chiave]]

                                    #Add ETER ID of Donor Selected

                                    imputation_test[diz_variabili_colonne_modifiche[variable_work][0]][i] = risultato_accoppiamento[chiave]


                                    #Add Distance Donor selected
                                    imputation_test[diz_variabili_colonne_modifiche[variable_work][2]][i] = risultato_distanze[chiave][1] 

                        except:

                            #print(anno)
                            pass






### Save the Results

imputation_test.to_excel(name_file_saved, sheet_name='Sheet_name_1',index = False)



#Now we read the beginning dataframe, inserting the beginning values that we have
#modified during our analysis.


imputation_test = pd.read_excel(name_file_saved)



### Results Analysis

### 2017 Donor Value is reported to an empty value.

#SubGroup of the beginning dataset
prova = imputation_test[imputation_test["Reference year"] == 2017].copy()





diz_var_FLAG ={ "Value Donor Enrolled":["ID Donor Enrolled", 
                                                              "Name Donor Enrolled", 
                                                              "Distance Donor Enrolled",
                                                             'Flag Enrolled'],
               
               
                                                  "Value Donor PhD Enrolled":["ID Donor PhD Enrolled",
                                                                "Name Donor PhD Enrolled", 
                                                                "Distance Donor PhD Enrolled",
                                                                "Flag PhD Enrolled"],
                                                                            
                                   
                                   "Value Donor PhD Graduates":["ID Donor PhD Graduates",
                                                                "Name Donor PhD Graduates", 
                                                                "Distance Donor PhD Graduates",
                                                                "Flag PhD Graduates"],              

                                   "Value Donor Graduates":["ID Donor Graduates", 
                                                              "Name Donor Graduates", 
                                                              "Distance Donor Graduates",
                                                             'Flag Graduates'],
                                   
                                   "Value Donor Academic Staff (FTE)":["ID Donor Academic Staff (FTE)", 
                                                                         "Name Donor Academic Staff (FTE)", 
                                                             "Distance Donor Academic Staff (FTE)",
                                                                        "Flag Academic Staff (FTE)"],
                                   
                                   
                                   
                                   "Value Donor Academic Staff (HC)":["ID Donor Academic Staff (HC)", 
                                                                        "Name Donor Academic Staff (HC)", 
                                                             "Distance Donor Academic Staff (HC)", "Flag Academic Staff (HC)" ],
                           
                                    "Value Donor Non Academic Staff (FTE)":["ID Donor Non Academic Staff (FTE)", 
                                                                        "Name Donor Non Academic Staff (FTE)", 
                                                             "Distance Donor Non Academic Staff (FTE)",
                                                                             "Flag Non Academic Staff (FTE)"],
                           
                                    "Value Donor Non Academic Staff (HC)":["ID Donor Non Academic Staff (HC)", 
                                                                        "Name Donor Non Academic Staff (HC)", 
                                                             "Distance Donor Non Academic Staff (HC)",
                                                                            "Flag Non Academic Staff (HC)"],
                                   
                                   "Value Donor Expenditure (EURO)":["ID Donor Expenditure (EURO)", 
                                                                       "Name Donor Expenditure (EURO)", 
                                                             "Distance Donor Expenditure (EURO)" ,
                                                                      "Flag Expenditure (EURO)"],
                                   
                                   "Value Donor Revenues (EURO)":["ID Donor Revenues (EURO)", 
                                                                    "Name Donor Revenues (EURO)", 
                                                                    "Distance Donor Revenues (EURO)",
                                                                   "Flag Revenues (EURO)"]
                                    }





#We work only with prova dataframe where there are Institutions from 2017
for i in prova.index:
    
    for var_studied in diz_var_FLAG:
        if (prova.loc[i][diz_var_FLAG[var_studied][3]] == "Donor" or prova.loc[i][diz_var_FLAG[var_studied][3]] == "Donor Rescaled" or prova.loc[i][diz_var_FLAG[var_studied][3]] == "Donor Relaxed Rescaled")  and prova.loc[i][var_studied] == "m":
            #Value
            imputation_test[var_studied][i] = "m"
            #Name
            imputation_test[diz_var_FLAG[var_studied][1]][i] = ""
            #ID
            imputation_test[diz_var_FLAG[var_studied][0]][i] = ""
            #Distance
            imputation_test[diz_var_FLAG[var_studied][2]][i] = ""
            #Flag
            imputation_test[diz_var_FLAG[var_studied][3]][i] = ""
    
imputation_test = imputation_test.replace(np.NaN, "" )



#Ordered Variable

if research_analysis == "1":

    var_ordered_toSave = ['ETER ID', 'Reference year', 'Institution Name',
           'Institution Category standardized', 'Institution Category - English',
           'English Institution Name', 'Country Code',
           'Distance education institution', 'Legal status',
           'Total students enrolled ISCED 5-7', 'Smooth Students Enrolled',  "Value Donor Enrolled"  , "Flag Enrolled", "Distance Donor Enrolled","ID Donor Enrolled",
           'Total graduates ISCED 5-7', 'Smooth Students Graduates', "Value Donor Graduates", "Flag Graduates", "Distance Donor Graduates", "ID Donor Graduates",
           'Total academic staff (FTE)', 'Smooth Academic Staff FTE',"Value Donor Academic Staff (FTE)","Flag Academic Staff (FTE)", "Distance Donor Academic Staff (FTE)","ID Donor Academic Staff (FTE)",
           
           'Total academic staff (HC)', 'Smooth Academic Staff HC', "Value Donor Academic Staff (HC)", "Flag Academic Staff (HC)", "Distance Donor Academic Staff (HC)", "ID Donor Academic Staff (HC)",
           'Number of non-academic  staff (FTE)', 'Smooth Non Academic Staff FTE', "Value Donor Non Academic Staff (FTE)", "Flag Non Academic Staff (FTE)", "Distance Donor Non Academic Staff (FTE)", "ID Donor Non Academic Staff (FTE)",
           'Number of non-academic staff (HC)', 'Smooth Non Academic Staff HC', "Value Donor Non Academic Staff (HC)","Flag Non Academic Staff (HC)", "Distance Donor Non Academic Staff (HC)","ID Donor Non Academic Staff (HC)",
           'Total Current expenditure (EURO)', 'Smooth Expenditure (EURO)', "Value Donor Expenditure (EURO)", "Flag Expenditure (EURO)","Distance Donor Expenditure (EURO)", "ID Donor Expenditure (EURO)",
           'Total Current revenues (EURO)', 'Smooth Revenues (EURO)', "Value Donor Revenues (EURO)", "Flag Revenues (EURO)", "Distance Donor Revenues (EURO)", "ID Donor Revenues (EURO)",
           'Total students enrolled at ISCED 8', 'Smooth PhD Enrolled', "Value Donor PhD Enrolled", "Flag PhD Enrolled", "Distance Donor PhD Enrolled", "ID Donor PhD Enrolled",
           'Total graduates at ISCED 8', 'Smooth PhD Graduates', "Value Donor PhD Graduates", "Flag PhD Graduates", "Distance Donor PhD Graduates","ID Donor PhD Graduates",
           'Research active institution', 'Reasearch Active Imputed',
           'FLAG Reasearch Active', 'p', 'pp(top 10)', 'mcs', 'mncs',
           'pp(industry)', 'pp(int collab)', 'pp(collab)',
           'Lowest degree delivered', 'Highest degree delivered']
    
elif research_analysis == "0":

    var_ordered_toSave = ['ETER ID', 'Reference year', 'Institution Name',
           'Institution Category standardized', 'Institution Category - English',
           'English Institution Name', 'Country Code',
           'Distance education institution', 'Legal status',
           'Total students enrolled ISCED 5-7', 'Smooth Students Enrolled',  "Value Donor Enrolled"  , "Flag Enrolled", "Distance Donor Enrolled","ID Donor Enrolled",
           'Total graduates ISCED 5-7', 'Smooth Students Graduates', "Value Donor Graduates", "Flag Graduates", "Distance Donor Graduates", "ID Donor Graduates",
           'Total academic staff (FTE)', 'Smooth Academic Staff FTE',"Value Donor Academic Staff (FTE)","Flag Academic Staff (FTE)", "Distance Donor Academic Staff (FTE)","ID Donor Academic Staff (FTE)",
           
           'Total academic staff (HC)', 'Smooth Academic Staff HC', "Value Donor Academic Staff (HC)", "Flag Academic Staff (HC)", "Distance Donor Academic Staff (HC)", "ID Donor Academic Staff (HC)",
           'Number of non-academic  staff (FTE)', 'Smooth Non Academic Staff FTE', "Value Donor Non Academic Staff (FTE)", "Flag Non Academic Staff (FTE)", "Distance Donor Non Academic Staff (FTE)", "ID Donor Non Academic Staff (FTE)",
           'Number of non-academic staff (HC)', 'Smooth Non Academic Staff HC', "Value Donor Non Academic Staff (HC)","Flag Non Academic Staff (HC)", "Distance Donor Non Academic Staff (HC)","ID Donor Non Academic Staff (HC)",
           'Total Current expenditure (EURO)', 'Smooth Expenditure (EURO)', "Value Donor Expenditure (EURO)", "Flag Expenditure (EURO)","Distance Donor Expenditure (EURO)", "ID Donor Expenditure (EURO)",
           'Total Current revenues (EURO)', 'Smooth Revenues (EURO)', "Value Donor Revenues (EURO)", "Flag Revenues (EURO)", "Distance Donor Revenues (EURO)", "ID Donor Revenues (EURO)",
           'Total students enrolled at ISCED 8', 'Smooth PhD Enrolled', "Value Donor PhD Enrolled", "Flag PhD Enrolled", "Distance Donor PhD Enrolled", "ID Donor PhD Enrolled",
           'Total graduates at ISCED 8', 'Smooth PhD Graduates', "Value Donor PhD Graduates", "Flag PhD Graduates", "Distance Donor PhD Graduates","ID Donor PhD Graduates"
           ]
    
imputation_test = imputation_test[var_ordered_toSave]

imputation_test.to_excel(name_file_saved, sheet_name='Sheet_name_1', index = False)


imputation_test = pd.read_excel(name_file_saved)

imputation_test = imputation_test.replace(np.NaN, "")
imputation_test.head()



# =============================================================================
# RATIOS - Correlated Variables
# =============================================================================

#The ratio could be the relationship between Graduates and Students during Reference years.
#(also PhD Graduates/PhD Students)

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

### Modification of the last results, updating the Ratios


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


imputation_test = addRatio(imputation_test, 'Value Donor PhD Graduates','Value Donor PhD Enrolled', "Ratio PhD Laureati/ PhD Iscritti")

#Ratios add only if the user is considering also "bibliometrics" variables.

if research_analysis == "1":
    
    imputation_test = addRatio(imputation_test, 'p','Value Donor Academic Staff (FTE)', "Ratio Publication/Academic Staff FTE")
    
    imputation_test = addRatio(imputation_test, 'p','Value Donor Academic Staff (HC)', "Ratio Publication/Academic Staff HC")
    
    imputation_test = addRatio(imputation_test, 'p','Value Donor PhD Enrolled', "Ratio Publication/ PhD Student")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (FTE)','Value Donor PhD Enrolled', "Academic Staff FTE/ PhD Student")

imputation_test = addRatio(imputation_test, 'Value Donor Academic Staff (HC)','Value Donor PhD Enrolled', "Academic Staff HC/ PhD Student")

imputation_test = addRatio(imputation_test, 'Value Donor PhD Enrolled','Value Donor Enrolled', "PhD Student/ Student")

imputation_test = addRatio(imputation_test, 'Value Donor PhD Graduates','Value Donor Graduates', "PhD Graduates/ Graduates")

imputation_test = addRatio(imputation_test, 'Value Donor PhD Enrolled','Value Donor Graduates', "PhD Student/ Graduates")





imputation_test = imputation_test.replace("Nada", "")
imputation_test.head()

## We save the Dataset after updating the results

imputation_test.to_excel(name_file_saved, sheet_name='Sheet_name_1',index = False)




#Now we have to bring back the starting values in the variables modify,
#at the beginning of the Donor Code. We need to observe the values from the dataSmooth file.

#Here we add old variables

var_from_Past = ['Total academic staff (HC)', 'Smooth Academic Staff HC',
       'Total academic staff (FTE)', 'Smooth Academic Staff FTE',
       'Number of non-academic  staff (FTE)', 'Smooth Non Academic Staff FTE',
       'Number of non-academic staff (HC)', 'Smooth Non Academic Staff HC',
       'Total Current expenditure (EURO)', 'Smooth Expenditure (EURO)',
       'Total Current revenues (EURO)', 'Smooth Revenues (EURO)',
       'Total students enrolled ISCED 5-7', 'Smooth Students Enrolled',
       'Total graduates ISCED 5-7', 'Smooth Students Graduates',
       'Total students enrolled at ISCED 8', 'Smooth PhD Enrolled',
       'Total graduates at ISCED 8', 'Smooth PhD Graduates']

#Reading Smooth Dataset (this was the starting dataset of our imputation)
with open('path_smooth.txt', encoding='utf-8') as f:
    path_smooth = f.read().replace('\n', '')

old_dataset = pd.read_excel(path_smooth + "./fileout_smooth_complete.xlsx")

for vv in tqdm(var_from_Past):
    imputation_test[vv] = old_dataset[vv]


imputation_test.to_excel(name_file_saved, sheet_name='Sheet_name_1',index = False)




imputation_test = pd.read_excel(name_file_saved)

imputation_test = imputation_test.replace(np.NaN, "")





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


missing_lavorati = ["c", "x", "xr", "s", "xc",1,0,2, "a"]




#Change and Detect errors within the Flag 

for i in tqdm(imputation_test.index):
    prova = imputation_test.loc[i].copy()
    #print(i)
    for var_worked in diz_variabili_colonne_modifiche:
        val_iniziale = prova[var_worked]
        val_smooth = prova[diz_variabili_colonne_modifiche[var_worked][4]]
        val_donato = prova[diz_variabili_colonne_modifiche[var_worked][5]]
        #flag_val = diz_variabili_colonne_modifiche[var_worked][3]
        #print(val_iniziale, val_smooth, val_donato)

                
        if val_iniziale in missing_lavorati and val_smooth != val_iniziale and val_smooth != "m":

            imputation_test[diz_variabili_colonne_modifiche[var_worked][3]][i] = "Smooth Change"
            
        if  val_iniziale in missing_lavorati and val_smooth == "m" and val_donato != val_iniziale and val_donato not in missing_lavorati:
            imputation_test[diz_variabili_colonne_modifiche[var_worked][3]][i] = "Donor Change"
            




#Change and Detect errors within the Flag 

for i in tqdm(imputation_test.index):
    prova = imputation_test.loc[i].copy()
    #print(i)
    for var_worked in diz_variabili_colonne_modifiche:
        
        val_iniziale = prova[var_worked]
        val_smooth = prova[diz_variabili_colonne_modifiche[var_worked][4]]
        val_donato = prova[diz_variabili_colonne_modifiche[var_worked][5]]
        #flag_val = diz_variabili_colonne_modifiche[var_worked][3]
        #print(val_iniziale, val_smooth, val_donato)

                
        if val_iniziale == "m" and val_smooth != val_iniziale:

            imputation_test[diz_variabili_colonne_modifiche[var_worked][3]][i] = "Smooth"
           


imputation_test = imputation_test.replace(np.NaN, "")

imputation_test.to_excel(name_file_saved, sheet_name='Sheet_name_1',index = False)



#Start the approtimation of Distance
imputation_test = pd.read_excel(name_file_saved)




#Extract all the Distance Columns
col_var_distance = [ var for var in imputation_test.columns if "Distance" in var and var != 'Distance education institution']
#Round them
imputation_test[col_var_distance] = imputation_test[col_var_distance].round(3)



#### Academic Staff FTE and Academic Staff HC different approximation
var_flag = {'Value Donor Academic Staff (HC)':'Flag Academic Staff (HC)', 
            'Value Donor Academic Staff (FTE)':'Flag Academic Staff (FTE)'}

for ind in tqdm(imputation_test.index):
    for var in var_flag:
        if imputation_test.loc[ind][var_flag[var]] != "":
            if imputation_test.loc[ind][var] != "m" and imputation_test.loc[ind][var] != "a":
                if imputation_test.loc[ind][var] < 10:
                    imputation_test[var][ind] = round(imputation_test.loc[ind][var],2)
                elif imputation_test.loc[ind][var] >= 10:
                    imputation_test[var][ind] = round(imputation_test.loc[ind][var],0)


imputation_test = imputation_test.replace(np.NaN, "")

imputation_test.to_excel(name_file_saved, sheet_name='Sheet_name_1',index = False)

print("Donor Imputation Relaxed Completed")
print()


