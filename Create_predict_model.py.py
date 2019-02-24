import pandas as pd  #pandas for data cleaning
import math #computation
import datetime
import re #regular expression

#getting current date and time
now = datetime.datetime.now()
#set universal e value to a vriable for ease of further computation
e = math.e

#getting the data from csv file and filling into a dataset
dataset = pd.read_csv('crop_production.csv',
                      usecols=['State_Name','Crop_Year','Season','Crop','Area','Production']).dropna()

# initializing weight matrix
#default weights are assign initially
V = [[1,1,1,1],[0,0,0,0],[1,0,0,1]] #weight_matrix_hidden_input
W = [1,1,1] #weight_matrix_output_hidden
Y = [0,0,0] #hidden layer output

#vocabulary to map each distinct word in CSV to an integer
state_vocab = {} #unique list of all states
crop_list = {} #unique list of all crops
season_list = {} #uniques list of all seasons

#variables for ease of further computation
e = math.e #set universal e value to a variable
error = 0  #error value
alpha = 0.1 #learning constant : it is set low to improve learning and avoid quick convergence

#Supervised neural networks:- Back Propagation
#running neural net with 1 hidden layer having 3 nodes
# 4 input nodes:- state,season, crop,crop_year
#1 output node:- conversion (production/area)
for iter in range (0,10): # retraining whole dataset for 10 times to increase accuraccy rate
    for i in range (0,10000) : # iterating over a dataset
        state_name = re.sub('[^a-zA-Z]', '', dataset['State_Name'][i]).lower() #getting state name and setting it to all lower with no spaces
        if state_name not in state_vocab: #populating unique state if not present in the map
            state_vocab[state_name] = len(state_vocab) #given each state integer value to be used while matrix computation in place of that state
        #all the strings values from dataset have been delt similar to how state values are handled
        #crop values retrieved and manipulated similar to state values
        crop = re.sub('[^a-zA-Z]', '', dataset['Crop'][i]).lower()
        if crop not in crop_list:
            crop_list[crop] = len(crop_list)
        #seasons value retrieved and manipulated similar to state values
        season = re.sub('[|]', ' ', dataset['Season'][i]).lower().replace(" ","")
        if season not in season_list:
            season_list[season] = len(season_list)
        #year of crop production is simply populated as it is already a numeric value, no alteration required
        crop_year = dataset['Crop_Year'][i]
        #calculating production per area
        #this factor will be used to assess the best crop that should be grown in a particular state at a particular season in a particular year
        conversion = dataset['Production'][i]/dataset['Area'][i]
        #input set: integer values of corresponding dataset values
        Z = [state_vocab[state_name],crop_list[crop],season_list[season],crop_year]
        #compute hidden layer
        for x in range (0,3): #since there are 3 nodes in hidden layer
            Y[x] = V[x][0]*state_vocab[state_name] + V[x][1]*crop_list[crop] + V[x][2]*season_list[season] + V[x][3]*crop_year
            Y[x] = math.sin(Y[x]) #to convert large result to a smaller more computable value
            Y[x] = 1/(1+e**(-Y[x])) #using bipolar continuous function to evaluate result
        
        #compute output layer
        O = 0 # initial output
        for x in range (0,3):
            O = O +  Y[x]*W[x]
            O = 1/(1+e**(-O))
        #output layer error computation and weight adjustment
        D = 1/(1+e**(-conversion))  #desired output
        error = error + (D-O)*(D-O)/2    #overall error
        del_op = (D - O)*(1 - O*O)/2     # error signal in the output
        for x in range (0,3):
            W[x] = W[x] + alpha*del_op*Y[x] #update weight matrix of output_hidden links
        #back propagated hidden layer error computation and adjustment
        del_h= [0,0,0]
        for x in range (0,3):
            del_h[x] = ((1-Y[x]*Y[x])*del_op*W[x])/2
            for m in range (0,4):
                V[x][m] = V[x][m] + alpha*del_h[x]*Z[m]
 
file = open("crop_predict_weight_matrix.txt", "a")    
for p in range (0,3):
        file.write(str(V[p]))
        file.write('\n')
file.write(str(W))       
file.close()

#commenting prediction algorithm
""" 
conversion_list = {}
for key in crop_list:
    Z = [state_vocab['andhrapradesh'],crop_list[key],season_list['kharif'],2019]
    for x in range (0,3):
        Y[x] = V[x][0]*Z[0] + V[x][1]*Z[1] + V[x][2]*Z[2] + V[x][3]*Z[3]
        Y[x] = math.tan(Y[x])
        Y[x] = 1/(1+e**(-Y[x]))
    O = 0
    for x in range (0,3):
        O = O +  Y[x]*W[x]
        O = 1/(1+e**(-O))   
    final_O =  math.log(O) - math.log(1-O)
    conversion_list[key]= final_O

best_choices = sorted(conversion_list.items(), key=lambda kv: kv[1], reverse= True)

print('best choices of crops for your are:')
counter = 1
for key in best_choices:
    print(key)
    if counter == 5:
        break
    counter = counter + 1    

"""