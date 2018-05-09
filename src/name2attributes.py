import csv
import pickle
import numpy as np

def main():
    path2data = "../data/"
    fin = open(path2data + "pokemon_id_name.csv")

    name2attr = {}

    type_set = set()

    HP_values = []
    Attack_values = []
    Defense_values = []
    SpAttack_values = []
    SpDefense_values = []
    Speed_values = []

    lines = fin.readlines()

    for line in lines[1:]:
        attributes = {}
        values = line.split(",")

        type_set.add(values[2])
        type_set.add(values[3])

        HP_values.append(int(values[5]))
        Attack_values.append(int(values[6]))
        Defense_values.append(int(values[7]))
        SpAttack_values.append(int(values[8]))
        SpDefense_values.append(int(values[9]))
        Speed_values.append(int(values[10]))

    HP_values = np.asarray(HP_values)
    Attack_values = np.asarray(Attack_values)
    Defense_values = np.asarray(Defense_values)
    SpAttack_values = np.asarray(SpAttack_values)
    SpDefense_values = np.asarray(SpDefense_values)
    Speed_values = np.asarray(Speed_values)

    print(type_set)
    print(len(type_set))

    print("Attack_max = " + str(Attack_values.max()))
    print("Defense_max = " + str(Defense_values.max()))
    print("SpAttack_max = " + str(SpAttack_values.max()))
    print("SpDefense_max = " + str(SpDefense_values.max()))
    print("Speed_max = " + str(Speed_values.max()))

    print("Attack_mean = " + str(Attack_values.mean()))
    print("Defense_mean = " + str(Defense_values.mean()))
    print("SpAttack_mean = " + str(SpAttack_values.mean()))
    print("SpDefense_mean = " + str(SpDefense_values.mean()))
    print("Speed_mean = " + str(Speed_values.mean()))

    print("Attack_std = " + str(Attack_values.std()))
    print("Defense_std = " + str(Defense_values.std()))
    print("SpAttack_std = " + str(SpAttack_values.std()))
    print("SpDefense_std = " + str(SpDefense_values.std()))
    print("Speed_std = " + str(Speed_values.std()))

    HP_max = HP_values.max()
    Attack_max = Attack_values.max()
    Defense_max = Defense_values.max()
    SpAttack_max = SpAttack_values.max()
    SpDefense_max = SpDefense_values.max()
    Speed_max = Speed_values.max()

    HP_mean = HP_values.mean()
    Attack_mean = Attack_values.mean()
    Defense_mean = Defense_values.mean()
    SpAttack_mean = SpAttack_values.mean()
    SpDefense_mean = SpDefense_values.mean()
    Speed_mean = Speed_values.mean()

    HP_std = HP_values.std()
    Attack_std = Attack_values.std()
    Defense_std = Defense_values.std()
    SpAttack_std = SpAttack_values.std()
    SpDefense_std = SpDefense_values.std()
    Speed_std = Speed_values.std()

    for line in lines[1:]:
        attributes = {}
        values = line.split(",")

        attributes["ID"] = int(values[0])
        # attributes["Name"] = values[1]
        attributes["Type1"] = values[2]
        attributes["Type2"] = values[3]
        attributes["Total"] = int(values[4])
        attributes["HP"] = (int(values[5])-HP_mean)/HP_std
        attributes["Attack"] = (int(values[6])-Attack_mean)/Attack_std
        attributes["Defense"] = (int(values[7])-Defense_mean)/Defense_std
        attributes["SpAttack"] = (int(values[8])-SpAttack_mean)/SpAttack_std
        attributes["SpDefense"] = (int(values[9])-SpDefense_mean)/SpDefense_std
        attributes["Speed"] = (int(values[10])-Speed_mean)/Speed_std
        attributes["Generation"] = int(values[11])
        attributes["Legendary"] = values[12]

        name2attr[values[1].lower()] = attributes

    with open(path2data + 'name2attributes.pkl', 'wb') as f:
        pickle.dump(name2attr, f, pickle.HIGHEST_PROTOCOL)






if __name__ == '__main__':
	main()
