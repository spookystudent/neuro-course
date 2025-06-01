# 1. Ошибка области видимости
def sum(a, b):
    c = a + b
    return c
print(c)  # Ошибка

# 2. Результаты ЕГЭ
math = int(input("Математика: "))
russian = int(input("Русский: "))
informatics = int(input("Информатика: "))

def passed(math, russian, informatics):
    total = math + russian + informatics
    if 120 <= total < 210:
        print('Хорошо')
    elif 210 <= total < 240:
        print('Очень хорошо')
    elif total >= 240:
        print('Отлично')
    else:
        print('Неуд')
    return total
print("Балл:", passed(math, russian, informatics))

# 3. Счетчик вызовов
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter
call_counter = make_counter()
print(call_counter(), call_counter(), call_counter())

# 4. Глобальная переменная
cake_count = 10
def modify_cake():
    global cake_count
    cake_count = 15
modify_cake()
print(cake_count)  # 15

# 5. Именованные аргументы
def final_price(price, discount=1):
    return price - price * discount / 100
print(final_price(1000, discount=5))  # 950.0
print(final_price(discount=10, price=1000))  # 900.0

# 6. Аргументы переменной длины
def print_args(*args, **kwargs):
    print("Позиционные:", args)
    print("Именованные:", kwargs)
print_args(1, 2, 3, name="Вася", age=25)