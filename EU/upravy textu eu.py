import textwrap

with open('eu.txt', 'r', encoding='utf-8') as f:
    text = f.read() # načte textový soubor

# zde jsou všechny unikátní znaky, které se v textu vyskytují
chars = sorted(list(set(text))) # vytvoří seznam unikátních znaků
vocab_size = len(chars) # počet unikátních znaků
print("----------------------------------------------------")
znaky = ''.join(chars) # spojení všech unikátních znaků do jednoho řetězce bez jakýchkoli oddělovačů
zalamovany_text = textwrap.fill(znaky, width=50)
# Tisk zalomeného textu
print(f"Model používá těchto {vocab_size} znaků:\n {zalamovany_text}") # tisk seznamu unikátních znaků

# oddeleni
# Původní řetězec
original_string = text

# Vypočítáme délku řetězce
length = len(original_string)

# Vypočítáme indexy pro odstranění 25 % z obou konců
cut_length = length // 4

# Získáme prostřední 50 % textu
middle_string = original_string[cut_length:-cut_length]

# Zápis prostředního řetězce do souboru
with open("eutexty.txt", 'w', encoding='utf-8') as file:
    file.write(middle_string)

print(f"Řetězec byl uložen do souboru")



