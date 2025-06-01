# 1. Проверка имени
name = input("Ваше имя: ")
print("Ура! Это же Вася!") if name == "Вася" else print("Привет,", name)

# 2. Проверка числа
num = int(input("Введите число: "))
print("Положительное" if num > 0 else "Отрицательное" if num < 0 else "Ноль")

# 3. Перевод в HEX
num = int(input("Введите число от 0 до 15: "))
print(hex(num)[2:].upper()) if 0 <= num <= 15 else print("Ошибка")

# 4. Вложенные условия
age = int(input("Ваш возраст: "))
if age >= 18:
    license = input("Есть права (да/нет)? ")
    if license == "да":
        print("Можно водить")
    else:
        print("Нужны права")
else:
    print("Сначала 18 лет")