# HOW TO USE
# call the function with "ecoScoreCalc(csv_file_path, search_string)"
# csv_file_path = path to file 
#     I used './fruit_carbon_and_water_footprint_data.csv'
# search_string = food you want to find, case insensitive
#     ex. = 'pineapple'
# returns wasfound, water, carbon, average
#         boolean, int,   int,   int
#         if not found, wasFound = False
#         if no water footprint data, water = -1
#         if no carbon footprint data, carbon = -1
#         if either carbon or water data isnt available, average = -1


import csv
import statistics
    
#function
def ecoScoreCalc(csv_file_path, search_string):
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        
        wasFound = False
        
        # assuming the first row contains headers, skip it if needed
        #headers = next(reader, None)
        
        for row_num, row in enumerate(reader, start=1):
            if search_string.lower() == row[0].lower():
                print(f"String '{search_string}' found.")
                wasFound = True
                break

        if not(wasFound):
            print(search_string, "was not found.")
            return wasFound, -1, -1, -1 #if it was found, water score, carbon score, average score
        
        
    #second part - find footprints
        waterValue = float(row[1])
        #print("The water footprint of a(n)", search_string, "is:", waterValue, "liters/kg")
    
        carbonValue = float(row[2])
        #print("The carbon footprint of a(n)", search_string, "is:", carbonValue, "kg/kg")

        
        
    #third part - find how eco friendly it is
        target_value = waterValue
        
        # extract the specified column
        column_values = [float(row[1]) for row in reader if float(row[1]) != 0]
        
        if (target_value == 0):
            print("Water footprint data not available")
            waterValue = -1
        else:
            mean_value = statistics.mean(column_values)

            # calculate and print the percentage difference for the target value
            waterValue = round((((waterValue - mean_value)/1.8109406130140314)+919)/300)
            if(waterValue > 5):
                waterValue = 5
            waterValue = 5 - waterValue
            print("The water eco-score a(n)", search_string, "is", waterValue, "out of 5")
            
        file.seek(0)  # reset the file pointer to the beginning
        target_value = float(carbonValue)

        if (target_value == 0):
            print("Carbon data not available")   
            carbonValue = -1  
        else: 
            # extract the specified column
            column_values = [float(row[2]) for row in reader if float(row[2]) != 0]
    
            mean_value = statistics.mean(column_values)
            
            ## calculate and print the percentage difference for each value
            #for index, value in enumerate(column_values, start=1):
            #    percentage_diff = calculate_carbon_score(value, mean_value, std_dev_value)
            #    if(percentage_diff > 5):
            #        percentage_diff = 5
            #    percentage_diff = 5 - percentage_diff
            #    print(f"Percentage Difference from Mean = {round(percentage_diff)}")

            # calculate and print the percentage difference for the target value
            carbonValue = round((carbonValue - mean_value + 0.47)*14.652014652)
            if(carbonValue > 5):
                carbonValue = 5
                carbonValue = 5 - carbonValue
            print("The carbon eco-score a(n)", search_string, "is", carbonValue, "out of 5")
            
        if(waterValue == -1 or carbonValue == -1):
            averageValue = -1
        else:
            averageValue = round((waterValue+carbonValue)/2)
            
        return wasFound, waterValue, carbonValue, averageValue
            


#TESTS
#csv_file_path = './fruit_carbon_and_water_footprint_data.csv'
#search_string = 'pineapple'
#found, water, carbon, average= ecoScoreCalc(csv_file_path, search_string)
#print(found, water, carbon, average)

#search_string = 'meow'
#found, water, carbon, average= find_ecoScoreCalc(csv_file_path, search_string)
#print(found, water, carbon, average)
#
#search_string = 'papaya'
#found, water, carbon, average= ecoScoreCalc(csv_file_path, search_string)
#print(found, water, carbon, average)
#
#search_string = 'ginger'
#found, water, carbon, average= ecoScoreCalc(csv_file_path, search_string)
#print(found, water, carbon, average)