import pandas as pd
import numpy as np
from sklearn import preprocessing

def gettingData(path, colsX, colsLabely):
    df = pd.read_csv(path)
    cols = ['RoofMaterial', 'ExteriorMaterial', 'FoundationMaterial']
    for col in cols:
        df[col] = pd.factorize(df[col])[0]
    X = df[colsX]
    y = df[colsLabely]
    return df, X, y

def normalizingData(X):
    return preprocessing.StandardScaler().fit(X).transform(X.astype(float))

def getCols():
    cols = ['DwellingType', 'OverallQuality', 
        'OverallConditions', 'YearBuilt', 'YearRemodel',
        'ExteriorQuality', 'ExteriorConditions', 'TotalBasementArea', 
        'HeatingQuality', 'HasAirConditioning', 'FirstFloorArea', 'SecondFloorArea', 'NumberOfBathrooms', 
        'NumberOfBedrooms', 'NumberOfKitchen', 'KitchenQuality', 'TotalNumberOfRooms', 
        'GarageYearBuilt', 'GarageNumberOfCars', 'GarageArea', 'GarageQuality', 'GarageCondition', 'PoolArea', 
        'PoolQuality', 'FenceQuality', 'MoSold', 'YrSold']
    
    return cols

def getFilters():
    filters = ['OverallQuality', 'OverallConditions', 'TotalNumberOfRooms', 'NumberOfBedrooms', 'NumberOfBathrooms',
    'GarageNumberOfCars', 'HouseStyleType', 'BuildingType', 'ExteriorQuality', 'HeatingQuality', 'HasAirConditioning']
    
    return filters

def getJsonFilters():
    filters = ['overallQuality', 'overallConditions', 'totalNumberOfRooms', 'numberOfBedrooms', 'numberOfBathrooms',
    'garageNumberOfCars', 'houseStyleType', 'buildingType', 'exteriorQuality', 'heatingQuality', 'hasAirConditioning']
    
    return filters
