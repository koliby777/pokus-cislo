with open('en.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

n = int(0.48*len(text))
cestina = text[:n]
anglictina = text[n:]
print(len(cestina))
print(len(anglictina))


"""
# Uložení češtiny do nového souboru
with open('C:/Users/sms88/Desktop/cestina.txt', 'w', encoding='utf-8') as soubor_cestina:
    soubor_cestina.write(cestina)

# Uložení angličtiny do nového souboru
with open('C:/Users/sms88/Desktop/anglictina.txt', 'w', encoding='utf-8') as soubor_anglictina:
    soubor_anglictina.write(anglictina)
"""

