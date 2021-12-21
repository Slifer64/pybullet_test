

if __name__ == '__main__':


    d = {1: "apple", 2:"bin"}

    d[4] = "spoon"

    if (d.has_key(5) == False):
        print("Addinig key 5 as potaeto!")
        d[5] = "potaeto"



    for key, value in d.items():
        print(key, ":", value)