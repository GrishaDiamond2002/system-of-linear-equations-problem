# Подключение библиотек
from prettytable import PrettyTable
import numpy as np 
from math import *
from decimal import *

# Задание исходных данных задачи
#Матрица коэффициентов системы линейных уравнений
A = np.array(
    [
        [4.405, 0.472, 0.395, 0.253],
        [0.227, 2.957, 0.342, 0.327],
        [0.419, 0.341, 3.238, 0.394],
        [0.325, 0.326, 0.401, 4.273],
    ]
)

#Вектор свободных коэффициентов системы
B = np.array([0.623, 0.072, 0.143, 0.065])

# функция округления числа до заданного количества значащих цифр
def round_to(num, digits=2) :
    num = Decimal (num)
    if num == 0:
        return 0
    scale = int(-floor(log10(abs(num -int(num))))) + digits - 1
    if scale < digits:
        scale = digits
    return round(num, scale)

# Функция ввода требуемой точности вычислений с проверкой корректности
def accuracy() :
    while True:
        try:
            toch = float(input("Введите точность вычислений = "))
            if 0 < toch < 1:
                break
            else:
                print (
                    "Некорректный ввод точности вычисления."
                    "Введите десятичное число в интервале от (0,1). "
                )
        except ValueError:
            print (
            "Некорректный ввод точности вычисления."
            "Введите десятичное число через точку."
        )
    return int(abs(log10(float(toch)))), toch

# Реализация метода простых итераций. 
def iter_method(A, b, k0):
    x = np.zeros(len(A))
    k = 0
    mytable = PrettyTable()
    mytable.field_names = ["k", "xl", "x2", "x3", "x4", "Hорма"]
    mytable.add_row(
        [
            k,
            round(x[0], kol_zn) ,
            round(x[1], kol_zn),
            round(x[2], kol_zn),
            round(x[3], kol_zn) ,
            "-",
        ]
    )
    while k < k0:
        x_old = np.copy(x)
        for i in range(len(A)):
            x[i] = sum([A[i][j]* x_old[j] for j in range(len(A[i])) if i != j]) + b[i]
        k += 1
        dx = max(abs(x_old - x))

        mytable.add_row(
            [
                k,
                round(x[0], kol_zn) ,
                round(x[1], kol_zn) ,
                round(x[2] , kol_zn) ,
                round(x[3], kol_zn),
                round(dx, kol_zn) ,
            ]
        )
    print (mytable)
    return x

#Реализация метода сопряжённых градиентов 
def CG(A,b,toch):
    #Иницилизация
    n = len(b)
    x = np.zeros(n)  
    r = b - A @ x    
    p = r.copy()     
    rs_old = np.dot(r, p)
    
    mytable = PrettyTable()
    mytable.field_names = ["k", "x1", "x2", "x3", "x4", "alfa", "beta", "Hорма"]

    #Итерационный процесс
    for k in range(1000):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)  
        x += alpha * p                 
        r -= alpha * Ap                
        rs_new = np.dot(r, r)          

        # Проверка критерия остановки
        norm = np.sqrt(rs_new)
        if norm < toch:
            break

        beta = rs_new / rs_old         
        p = r + beta * p              
        rs_old = rs_new

        mytable.add_row(
            [
                k+1,
                round(x[0], kol_zn) ,
                round(x[1], kol_zn) ,
                round(x[2] , kol_zn),
                round(x[3], kol_zn),
                round(alpha,kol_zn),
                round(beta,kol_zn),
                round(norm, kol_zn)
            ]
            )

    print (mytable)
    return x, k


#Реализация метода минимальных невязок
def minNevaz(A,b,toch):
    n = len(b)
    x = np.zeros(n)  
    r = b - A @ x
    q = A @ r

    mytable = PrettyTable()
    mytable.field_names = ["k", "x1", "x2", "x3", "x4", "tetta", "Hорма"]

    for k in range(1000):

        tetta = np.dot(r, r) / np.dot(r, q)
        x = x + tetta * r
        r_new = r - tetta * q


        # Проверка критерия остановки
        norm = np.linalg.norm(r_new)
        if norm < toch:
            break

        # Обновление параметров
        q = A @ r_new
        r = r_new

        mytable.add_row(
                    [
                        k+1,
                        round(x[0], kol_zn) ,
                        round(x[1], kol_zn) ,
                        round(x[2] , kol_zn),
                        round(x[3], kol_zn),
                        round(tetta,kol_zn),
                        round(norm, kol_zn)
                    ]
                    )

    print (mytable)
    return x, k 


print("Решение СЛАУ вариационными методами")

print ("\nМатрица А")
for i in range(len(A)):
        for j in range(len(A[i])):
            print((A[i] [j]),  " ", end="")
        print ("")

# Вызов функции задания точности вычислений
kol_zn, toch = accuracy()

# Основная функция вывода решения СЛАУ
def main():
    
    #Вывод матрицы А
    
    print("Метод простых истераций (МПИ). Метод Якоби.")
    #Вывод обратной матрицы
    print ("\nОбратная матрица")
    A_inv = np.linalg.inv(A)
    for i in range(len(A_inv)):
        for j in range(len(A_inv[i])):
            print(round(A_inv[i] [j], kol_zn)," ", end="")
        print ("")
    #Вычисление нормы матрицы А
    A_norm = max(list([round(sum(abs(A[i])), kol_zn) for i in range(len(A))]))
    print(f"\nHорма матрицы ||A|| = {A_norm}")

    # Вычисление нормы обратной матрицы А
    A_inv_norm = max(list([round(sum(abs(A_inv[i])), kol_zn) for i in range(len(A_inv))]))
    print(f"Норма обратной матрицы = {A_inv_norm}")

    # Вычислние нормы вектора b
    B_norm = max(B, key=abs)
    print(f"Норма вектора ||b|| = {B_norm}")

    # Преобразование системы для применения вариционных методов решения СЛАУ
    print("\преобразованная система, необходимая для применения метода Якоби")
    new_A = A.copy()
    new_B = B.copy()
    for i in range(len(A)):
        numi = A[i][i]
        for j in range(len(A[i])):
            new_A[i][j] /= numi
            new_A[i][j] = round(new_A[i][j], kol_zn)
            print(new_A[i][j]," ",end = "")
        new_B[i] /= numi
        new_B[i] = round(new_B[i], kol_zn)
        print (new_B[i])
        
    for i in range(len(new_A)):
        new_A[i][i] = 0
        
    print("\nВектор С")
    print (new_B)
    print("\nMатрица B")
    print (-new_A)

    # Проверка условия сходимости МПИ
    print("\nпроверка условия сходимости МПИ")
    B_new_norm = max(list([round(sum(abs(new_A[i])), kol_zn) for i in range(len(new_A))]))
    if B_new_norm < 1:
        print(f"Hорма ||B|| = {B_new_norm} < 1 =>"
          "\nусловия теоремы о сходимости МПИ выполнены")
        c = max(new_B, key=abs)
        k0 = log(((toch) * (1 - B_new_norm) / c), e) / log(B_new_norm, e)
        print(f"\n Количество итерационных шагов для метода Якоби k0={ceil(k0)}")
    else:
        print(f"Условия теоремы о сходимости МПИ не выполнены")
        k0 = 0
        
    # Вызов функции метода Якоби
    print ("\nМетод простых итераций")
    iter = iter_method(-new_A, new_B, k0)
    print (
    f"x1 = {round(iter[0],kol_zn-1)} x2 = {round(iter[1],kol_zn-1)} "
    f"x3 = {round(iter[2],kol_zn-1)} x4 = {round(iter[3],kol_zn-1)}"
    )

    print("\n Метод сопряжённых градиентов")

    #Проверка симметичности и её преобразование
    print("Проверим симметричность матрицы для решения методом сопряжённых градиентов")
    print("Матрица A:")
    transA = np.transpose(A)
    #Вывод матрицы А
    for i in range(len(A)):
            for j in range(len(A[i])):
                print(round(A[i] [j], kol_zn), " ", end="")
            print ("")
    if np.allclose(A,transA) == True:
        print("\nМатрица А симметрична")
    else:
        print("Матрица А несимметрична.")
        NewSimmetricA = (A+transA)/2
        print("\nНовая симметричная матрица:")
        #Вывод матрицы
        for i in range(len(NewSimmetricA)):
            for j in range(len(NewSimmetricA[i])):
                print(round(NewSimmetricA[i] [j], kol_zn), " ", end="")
            print ("")

    #Вызов метода сопряжённых градиентов
    print("\n Метод сопряжённых градиентов")
    CG_main, iteration = CG(A,B,toch)
    print(
    f"x1 = {round(CG_main[0],kol_zn-1)} x2 = {round(CG_main[1],kol_zn-1)} "
    f"x3 = {round(CG_main[2],kol_zn-1)} x4 = {round(CG_main[3],kol_zn-1)} "
    f"итераций = {round(iteration,kol_zn-1)}"
    )

    #Вызов метода минимальных невязок
    print("\n Метод минимальных невязок")
    MN_main,iteration = minNevaz(A,B,toch)
    print(
    f"x1 = {round(MN_main[0],kol_zn-1)} x2 = {round(MN_main[1],kol_zn-1)} "
    f"x3 = {round(MN_main[2],kol_zn-1)} x4 = {round(MN_main[3],kol_zn-1)} "
    f"итераций = {round(iteration,kol_zn-1)}"
    )
    
main()
