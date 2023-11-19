def testovani():
    print("T E S T O V Á N Í   Č Í S E L")
    print("(pro ukončení tesování zadej místo čísla: konec)")
    # Cyklus pokračuje, dokud uživatel nezadá slovo konec
    while True:
        a = input("Zadej reálné číslo: ")
        try:
            # zkusí převést zadaný řetězec na reálné číslo 
            a = float(a)
        # výjimka: špatná hodnota - zadaný řetězec není reálné číslo, např. různé nečíselné znaky, včetně slova konec
        except ValueError:
            if a == "konec":
                # aby si uživatel event. mohl naposledy prohlédnout výsledky
                input("...a ještě jednou stiskni Enter ...")
                # cyklus končí
                break   
            else:
                print("To není číslo !!!")
            # pokračuje na začátek cyklu
            continue
        if a > 0:
            print("kladné")
        elif a < 0:
            print("záporné")
        else:
            print("nula")

if __name__ == "__main__":
    testovani()



